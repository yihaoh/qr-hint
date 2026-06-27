"""
Subquery Rewriter Module

Transforms SQL queries containing subqueries (EXISTS, ANY/SOME, ALL) into equivalent
single queries without subqueries.

Transformation Rules:
1. EXISTS (correlated subquery) -> DISTINCT + JOIN
   SELECT * FROM T1 WHERE EXISTS (SELECT * FROM T2 WHERE T2.id = T1.id)
   => SELECT DISTINCT T1.* FROM T1, T2 WHERE T2.id = T1.id

2. col OP SOME (subquery) -> DISTINCT + JOIN with comparison
   SELECT * FROM T1 WHERE T1.col > SOME (SELECT col FROM T2 WHERE cond)
   => SELECT DISTINCT T1.* FROM T1, T2 WHERE T1.col > T2.col AND cond

3. col = SOME (subquery) is equivalent to IN
   SELECT * FROM T1 WHERE T1.col = SOME (SELECT col FROM T2)
   => SELECT DISTINCT T1.* FROM T1, T2 WHERE T1.col = T2.col

4. Nested subqueries are handled recursively

Limitations:
- NOT EXISTS cannot be fully converted without LEFT JOIN support
- ALL subqueries are not supported (require aggregation or NOT EXISTS)
- Assumes subqueries are correlated subqueries
"""

import json
from copy import deepcopy
from .global_var_beers import default_analyzer


class ExistsRewriter:
    """
    Rewrites queries with EXISTS subqueries into equivalent single queries with DISTINCT.
    """

    def __init__(self, query: str, analyzer=default_analyzer):
        """
        Initialize the rewriter.

        Args:
            query: Original SQL query string
            analyzer: SQL analyzer (Calcite-based)
        """
        self.original_query = query
        self.analyzer = analyzer
        self.data = json.loads(str(self.analyzer.analyzeToJson(query)))

        if "error" in self.data:
            raise RuntimeError(f"SQL syntax error: {self.data.get('message', 'Unknown error')}")

        self.xtree = self.data["xtree"]
        self.has_exists = False
        self.rewritten_query = None

        # Tables to be added to FROM clause
        self.tables_to_add = []

        # Track all table names/aliases used in the query (for conflict detection)
        self.used_table_names = set()

    def _collect_table_names(self, from_node, names=None) -> set:
        """
        Collect all table names and aliases from a FROM clause.

        Args:
            from_node: FROM clause node
            names: Set to accumulate names (for recursion)

        Returns:
            set: All table names and aliases
        """
        if names is None:
            names = set()

        if not from_node or not isinstance(from_node, dict):
            return names

        node_type = from_node.get("type")

        if node_type == "XTableRefNode":
            table_name = from_node.get("name", "")
            if table_name:
                names.add(table_name)

        elif node_type == "XTableRenameNode":
            alias = from_node.get("new_name", "")
            if alias:
                names.add(alias)
            else:
                table_name = from_node.get("operand", {}).get("name", "")
                if table_name:
                    names.add(table_name)

        elif node_type == "XJoinNode":
            self._collect_table_names(from_node.get("left"), names)
            self._collect_table_names(from_node.get("right"), names)

        return names

    def _generate_unique_alias(self, base_name: str) -> str:
        """
        Generate a unique alias for a table that conflicts with existing names.

        Args:
            base_name: Original table name

        Returns:
            str: Unique alias like "Employees_1", "Employees_2", etc.
        """
        counter = 1
        while True:
            new_alias = f"{base_name}_{counter}"
            if new_alias not in self.used_table_names:
                self.used_table_names.add(new_alias)
                return new_alias
            counter += 1

    def _get_table_identifier(self, from_node) -> str:
        """
        Get the identifier (alias or table name) used to reference a table.
        """
        if not from_node or not isinstance(from_node, dict):
            return ""

        node_type = from_node.get("type")

        if node_type == "XTableRefNode":
            return from_node.get("name", "")
        elif node_type == "XTableRenameNode":
            return from_node.get("new_name", "") or from_node.get("operand", {}).get("name", "")

        return ""

    def _rename_table_references(self, node, old_name: str, new_name: str):
        """
        Recursively rename all column references from old_name to new_name.

        Args:
            node: AST node to process
            old_name: Original table name/alias
            new_name: New alias to use
        """
        if not node:
            return

        if isinstance(node, dict):
            node_type = node.get("type")

            # Handle XFieldRefNode (used in some contexts)
            if node_type == "XFieldRefNode":
                table_ref = node.get("table_name", "")
                if table_ref == old_name:
                    node["table_name"] = new_name
                    if "sql_string" in node:
                        col_name = node.get("field_name", "")
                        node["sql_string"] = f"{new_name}.{col_name}"

            # Handle XColumnRefNode (used in subqueries)
            elif node_type == "XColumnRefNode":
                sql_string = node.get("sql_string", "")
                # Check if sql_string starts with "old_name."
                if sql_string.startswith(f"{old_name}."):
                    col_name = sql_string[len(old_name) + 1:]
                    node["sql_string"] = f"{new_name}.{col_name}"

            # Handle XBasicCallNode - update sql_string for the whole expression
            elif node_type == "XBasicCallNode":
                if "sql_string" in node:
                    # Replace table references in sql_string
                    # Use word boundary to avoid partial replacements
                    import re
                    pattern = rf'\b{re.escape(old_name)}\.'
                    node["sql_string"] = re.sub(pattern, f"{new_name}.", node["sql_string"])

            # Recursively process all child nodes
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    self._rename_table_references(value, old_name, new_name)

        elif isinstance(node, list):
            for item in node:
                self._rename_table_references(item, old_name, new_name)

    def _add_alias_to_table(self, from_node, alias: str):
        """
        Wrap a table reference with an alias.

        Returns:
            dict: New XTableRenameNode with alias
        """
        if from_node.get("type") == "XTableRefNode":
            return {
                "type": "XTableRenameNode",
                "operand": from_node,
                "new_name": alias
            }
        elif from_node.get("type") == "XTableRenameNode":
            from_node["new_name"] = alias
            return from_node
        return from_node

    # Operators that indicate subqueries we can rewrite
    SUPPORTED_SUBQUERY_OPS = ["EXISTS", "= SOME", "> SOME", "< SOME", ">= SOME", "<= SOME", "<> SOME"]
    UNSUPPORTED_SUBQUERY_OPS = ["NOT EXISTS", "> ALL", "< ALL", ">= ALL", "<= ALL", "= ALL", "<> ALL"]

    def contains_exists(self, xnode=None) -> bool:
        """
        Check if the query contains EXISTS subqueries.

        Returns:
            bool: True if query contains EXISTS, False otherwise
        """
        return self.contains_subquery(xnode, ["EXISTS", "NOT EXISTS"])

    def contains_some_any(self, xnode=None) -> bool:
        """
        Check if the query contains ANY/SOME subqueries.

        Returns:
            bool: True if query contains ANY/SOME, False otherwise
        """
        some_ops = ["= SOME", "> SOME", "< SOME", ">= SOME", "<= SOME", "<> SOME"]
        return self.contains_subquery(xnode, some_ops)

    def contains_all(self, xnode=None) -> bool:
        """
        Check if the query contains ALL subqueries.

        Returns:
            bool: True if query contains ALL, False otherwise
        """
        all_ops = ["> ALL", "< ALL", ">= ALL", "<= ALL", "= ALL", "<> ALL"]
        return self.contains_subquery(xnode, all_ops)

    def contains_subquery(self, xnode=None, operators=None) -> bool:
        """
        Check if the query contains any of the specified subquery operators.

        Args:
            xnode: Node to check (defaults to self.xtree)
            operators: List of operator names to check for

        Returns:
            bool: True if query contains any of the operators, False otherwise
        """
        if xnode is None:
            xnode = self.xtree
        if operators is None:
            operators = self.SUPPORTED_SUBQUERY_OPS + self.UNSUPPORTED_SUBQUERY_OPS

        if isinstance(xnode, dict):
            # Check current node
            if xnode.get("type") == "XBasicCallNode":
                op_name = xnode.get("operator_name", "")
                if op_name in operators:
                    return True

            # Recursively check all child nodes
            for key, value in xnode.items():
                if isinstance(value, (dict, list)):
                    if self.contains_subquery(value, operators):
                        return True

        elif isinstance(xnode, list):
            for item in xnode:
                if self.contains_subquery(item, operators):
                    return True

        return False

    def contains_any_subquery(self, xnode=None) -> bool:
        """
        Check if the query contains any type of subquery (EXISTS, SOME, ALL).

        Returns:
            bool: True if query contains any subquery, False otherwise
        """
        return self.contains_subquery(xnode)

    def rewrite(self) -> str:
        """
        Execute the rewrite and return the transformed SQL.

        Returns:
            str: Rewritten SQL if subquery was found, otherwise original SQL
        """
        # Check if any rewritable subquery exists
        has_exists = self.contains_exists()
        has_some = self.contains_some_any()

        if not has_exists and not has_some:
            return self.original_query

        self.has_exists = has_exists or has_some
        self.tables_to_add = []

        # Deep copy xtree for modification
        new_xtree = deepcopy(self.xtree)

        # Collect existing table names from the outer query
        self.used_table_names = self._collect_table_names(new_xtree.get("from_expr"))

        # Recursively rewrite all subqueries (EXISTS and SOME)
        self._rewrite_subqueries_in_tree(new_xtree, new_xtree)

        # Add DISTINCT to eliminate duplicates from join
        new_xtree["is_distinct"] = True

        # Add collected tables to FROM clause
        for table_from in self.tables_to_add:
            self._add_table_to_from(new_xtree, table_from)

        # Generate SQL from modified xtree
        self.rewritten_query = self._generate_sql_from_xtree(new_xtree)

        return self.rewritten_query

    def _rewrite_subqueries_in_tree(self, xnode, root_xtree, parent=None, key=None):
        """
        Recursively rewrite all subqueries (EXISTS and SOME/ANY) in the xtree.

        Args:
            xnode: Current node being processed
            root_xtree: Root xtree (for adding tables)
            parent: Parent node
            key: Key of current node in parent
        """
        if not isinstance(xnode, dict):
            if isinstance(xnode, list):
                for i, item in enumerate(xnode):
                    self._rewrite_subqueries_in_tree(item, root_xtree, xnode, i)
            return

        # Check if this is a subquery-related call
        if xnode.get("type") == "XBasicCallNode":
            op_name = xnode.get("operator_name", "")

            if op_name == "EXISTS":
                # Get the subquery
                subquery = xnode.get("operands", [{}])[0]
                if subquery.get("type") == "XSelectNode":
                    # Extract subquery information
                    replacement = self._process_exists_subquery(subquery, root_xtree)

                    # Replace the EXISTS node
                    if parent is not None and key is not None:
                        parent[key] = replacement
                    return

            elif op_name == "NOT EXISTS":
                print(f"Warning: NOT EXISTS cannot be fully converted without LEFT JOIN support")

            elif op_name in self.SUPPORTED_SUBQUERY_OPS and op_name != "EXISTS":
                # Handle SOME/ANY subqueries: = SOME, > SOME, < SOME, etc.
                operands = xnode.get("operands", [])
                if len(operands) == 2:
                    left_operand = operands[0]  # The column being compared
                    subquery = operands[1]  # The subquery

                    if subquery.get("type") == "XSelectNode":
                        # Extract the comparison operator (e.g., ">", "=", "<")
                        cmp_op = op_name.split()[0]  # ">" from "> SOME"

                        replacement = self._process_some_subquery(
                            left_operand, subquery, cmp_op, root_xtree
                        )

                        # Replace the SOME node
                        if parent is not None and key is not None:
                            parent[key] = replacement
                        return

            elif op_name in self.UNSUPPORTED_SUBQUERY_OPS:
                print(f"Warning: {op_name} subqueries are not supported")

        # Recursively process child nodes
        for k, v in list(xnode.items()):
            if isinstance(v, (dict, list)):
                self._rewrite_subqueries_in_tree(v, root_xtree, xnode, k)

    # Keep old method name for backward compatibility
    def _rewrite_exists_in_tree(self, xnode, root_xtree, parent=None, key=None):
        """Backward compatible alias for _rewrite_subqueries_in_tree"""
        return self._rewrite_subqueries_in_tree(xnode, root_xtree, parent, key)

    def _process_exists_subquery(self, subquery, root_xtree):
        """
        Process an EXISTS subquery.

        Returns:
            Node to replace EXISTS (the subquery's WHERE condition)
        """
        # First recursively process any nested subqueries
        self._rewrite_subqueries_in_tree(subquery, root_xtree)

        # Extract subquery's FROM (tables)
        subquery_from = subquery.get("from_expr")
        subquery_where = subquery.get("where_cond")

        if subquery_from:
            subquery_from = deepcopy(subquery_from)
            subquery_where = deepcopy(subquery_where) if subquery_where else None

            # Handle table name conflicts
            subquery_from, subquery_where = self._resolve_table_conflicts(
                subquery_from, subquery_where
            )

            self.tables_to_add.append(subquery_from)

        if subquery_where:
            return subquery_where
        else:
            # If no WHERE, return TRUE (EXISTS is always true for non-empty tables)
            return self._create_true_node()

    def _process_some_subquery(self, left_operand, subquery, cmp_op, root_xtree):
        """
        Process a SOME/ANY subquery.

        Transforms: col OP SOME (SELECT x FROM T WHERE cond)
        Into: col OP T.x AND cond (with T added to FROM)

        Args:
            left_operand: The left side of the comparison (e.g., Frequents.times_a_week)
            subquery: The subquery node
            cmp_op: The comparison operator (e.g., ">", "=", "<")
            root_xtree: Root xtree for context

        Returns:
            Node to replace the SOME expression (comparison + subquery WHERE)
        """
        # First recursively process any nested subqueries
        self._rewrite_subqueries_in_tree(subquery, root_xtree)

        # Extract subquery's FROM and WHERE
        subquery_from = subquery.get("from_expr")
        subquery_where = subquery.get("where_cond")
        subquery_select = subquery.get("select_exprs", [])

        # Get the column being selected in the subquery (first SELECT expression)
        if not subquery_select:
            raise RuntimeError("SOME/ANY subquery must have a SELECT expression")

        subquery_col = deepcopy(subquery_select[0])

        if subquery_from:
            subquery_from = deepcopy(subquery_from)
            subquery_where = deepcopy(subquery_where) if subquery_where else None

            # Handle table name conflicts - also update subquery_col references
            old_table_names = self._collect_table_names(subquery_from)
            subquery_from, subquery_where = self._resolve_table_conflicts(
                subquery_from, subquery_where
            )
            # Also update references in subquery_col
            new_table_names = self._collect_table_names(subquery_from)

            # If table names changed, update subquery_col
            for old_name in old_table_names:
                if old_name not in new_table_names:
                    # Find the new alias
                    for new_name in new_table_names:
                        if new_name.startswith(old_name + "_"):
                            self._rename_table_references(subquery_col, old_name, new_name)
                            break

            self.tables_to_add.append(subquery_from)

        # Create comparison node: left_operand OP subquery_col
        comparison_node = {
            "type": "XBasicCallNode",
            "operator_name": cmp_op,
            "operands": [deepcopy(left_operand), subquery_col],
            "sql_string": f"{left_operand.get('sql_string', '')} {cmp_op} {subquery_col.get('sql_string', '')}"
        }

        # Combine with subquery's WHERE if exists
        if subquery_where:
            return {
                "type": "XBasicCallNode",
                "operator_name": "AND",
                "operands": [comparison_node, subquery_where],
                "sql_string": f"({comparison_node['sql_string']} AND {subquery_where.get('sql_string', '')})"
            }
        else:
            return comparison_node

    def _resolve_table_conflicts(self, from_node, where_node):
        """
        Check and resolve table name conflicts between subquery and outer query.

        Args:
            from_node: Subquery's FROM clause
            where_node: Subquery's WHERE clause

        Returns:
            tuple: (modified_from_node, modified_where_node)
        """
        if not from_node:
            return from_node, where_node

        node_type = from_node.get("type")

        if node_type == "XTableRefNode":
            # Simple table reference
            table_name = from_node.get("name", "")
            if table_name in self.used_table_names:
                # Conflict! Generate unique alias
                new_alias = self._generate_unique_alias(table_name)
                # Rename references in WHERE clause
                if where_node:
                    self._rename_table_references(where_node, table_name, new_alias)
                # Add alias to table
                from_node = self._add_alias_to_table(from_node, new_alias)
            else:
                self.used_table_names.add(table_name)

        elif node_type == "XTableRenameNode":
            # Table with alias
            alias = from_node.get("new_name", "")
            original_table = from_node.get("operand", {}).get("name", "")
            identifier = alias or original_table

            if identifier in self.used_table_names:
                # Conflict! Generate unique alias
                new_alias = self._generate_unique_alias(original_table)
                # Rename references in WHERE clause
                if where_node:
                    self._rename_table_references(where_node, identifier, new_alias)
                # Update alias
                from_node["new_name"] = new_alias
            else:
                self.used_table_names.add(identifier)

        elif node_type == "XJoinNode":
            # JOIN node: process both sides
            left = from_node.get("left")
            right = from_node.get("right")

            if left:
                new_left, where_node = self._resolve_table_conflicts(left, where_node)
                from_node["left"] = new_left
            if right:
                new_right, where_node = self._resolve_table_conflicts(right, where_node)
                from_node["right"] = new_right

        return from_node, where_node

    def _add_table_to_from(self, xtree, table_from):
        """
        Add a table to the xtree's FROM clause using comma join.
        """
        main_from = xtree.get("from_expr")

        if main_from:
            # Create new JOIN node (COMMA JOIN)
            new_join = {
                "type": "XJoinNode",
                "join_type": "COMMA",
                "left": main_from,
                "right": table_from
            }
            xtree["from_expr"] = new_join
        else:
            xtree["from_expr"] = table_from

    def _create_true_node(self):
        """Create a node representing TRUE literal."""
        return {
            "type": "XLiteralNode",
            "data_type": "BOOLEAN",
            "sql_string": "TRUE",
            "value": True
        }

    def _generate_sql_from_xtree(self, xtree) -> str:
        """
        Generate SQL string from modified xtree.
        """
        parts = []

        # SELECT clause
        select_keyword = "SELECT DISTINCT" if xtree.get("is_distinct") else "SELECT"
        select_exprs = xtree.get("select_exprs", [])
        select_cols = [self._expr_to_sql(expr) for expr in select_exprs]
        parts.append(f"{select_keyword} {', '.join(select_cols) if select_cols else '*'}")

        # FROM clause
        from_expr = xtree.get("from_expr")
        if from_expr:
            from_sql = self._from_to_sql(from_expr)
            parts.append(f"FROM {from_sql}")

        # WHERE clause
        where_cond = xtree.get("where_cond")
        if where_cond:
            where_sql = self._expr_to_sql(where_cond)
            parts.append(f"WHERE {where_sql}")

        # GROUP BY clause
        group_by = xtree.get("group_by_exprs")
        if group_by:
            group_cols = [self._expr_to_sql(expr) for expr in group_by]
            parts.append(f"GROUP BY {', '.join(group_cols)}")

        # HAVING clause
        having = xtree.get("having_cond")
        if having:
            having_sql = self._expr_to_sql(having)
            parts.append(f"HAVING {having_sql}")

        return " ".join(parts)

    def _from_to_sql(self, from_node) -> str:
        """Convert FROM node to SQL string."""
        if not from_node:
            return ""

        node_type = from_node.get("type")

        if node_type == "XTableRefNode":
            return from_node.get("name", "")

        elif node_type == "XTableRenameNode":
            table_name = from_node.get("operand", {}).get("name", "")
            alias = from_node.get("new_name", "")
            return f"{table_name} AS {alias}" if alias else table_name

        elif node_type == "XJoinNode":
            left = self._from_to_sql(from_node.get("left"))
            right = self._from_to_sql(from_node.get("right"))
            join_type = from_node.get("join_type", "COMMA")

            if join_type == "COMMA":
                return f"{left}, {right}"
            else:
                return f"{left} {join_type} JOIN {right}"

        return from_node.get("sql_string", "")

    def _expr_to_sql(self, expr_node) -> str:
        """Convert expression node to SQL string."""
        if not expr_node:
            return ""

        node_type = expr_node.get("type")

        # For compound expressions, don't use sql_string directly; need recursive processing
        if node_type == "XBasicCallNode":
            op_name = expr_node.get("operator_name", "")
            operands = expr_node.get("operands", [])

            # Logical operators - process recursively
            if op_name in ["AND", "OR"]:
                op_sqls = [self._expr_to_sql(op) for op in operands]
                return f"({f' {op_name} '.join(op_sqls)})"

            elif op_name == "NOT":
                return f"NOT ({self._expr_to_sql(operands[0])})"

            # Comparison operators
            elif op_name in ["=", "<>", "<", ">", "<=", ">="]:
                left = self._expr_to_sql(operands[0]) if operands else ""
                right = self._expr_to_sql(operands[1]) if len(operands) > 1 else ""
                return f"{left} {op_name} {right}"

            # Arithmetic operators
            elif op_name in ["+", "-", "*", "/"]:
                left = self._expr_to_sql(operands[0]) if operands else ""
                right = self._expr_to_sql(operands[1]) if len(operands) > 1 else ""
                return f"({left} {op_name} {right})"

            # Aggregate functions
            elif op_name in ["SUM", "AVG", "COUNT", "MAX", "MIN"]:
                arg = self._expr_to_sql(operands[0]) if operands else "*"
                return f"{op_name}({arg})"

            # LIKE operator
            elif op_name == "LIKE":
                left = self._expr_to_sql(operands[0]) if operands else ""
                right = self._expr_to_sql(operands[1]) if len(operands) > 1 else ""
                return f"{left} LIKE {right}"

            # Other operators: use sql_string
            return expr_node.get("sql_string", "")

        # Simple expressions can use sql_string directly
        if "sql_string" in expr_node:
            return expr_node["sql_string"]

        return str(expr_node)


def rewrite_exists_query(query: str, analyzer=default_analyzer) -> str:
    """
    Convenience function: Rewrite queries containing EXISTS.

    Args:
        query: Original SQL query
        analyzer: SQL analyzer

    Returns:
        str: Rewritten SQL if EXISTS was found, otherwise original SQL
    """
    rewriter = ExistsRewriter(query, analyzer)
    return rewriter.rewrite()


# Test cases
if __name__ == "__main__":
    # Test queries
    test_queries = [
        # Simple EXISTS
        "SELECT * FROM Frequents WHERE EXISTS (SELECT * FROM Likes WHERE Likes.drinker = Frequents.drinker)",

        # No EXISTS
        "SELECT * FROM Frequents WHERE drinker = 'Alice'",

        # EXISTS with additional conditions
        "SELECT drinker FROM Frequents WHERE bar = 'James Joyce Pub' AND EXISTS (SELECT * FROM Likes WHERE Likes.drinker = Frequents.drinker AND beer = 'Bud')",

        # Nested EXISTS
        "SELECT * FROM Frequents WHERE EXISTS (SELECT * FROM Likes WHERE Likes.drinker = Frequents.drinker AND EXISTS (SELECT * FROM Serves WHERE Serves.beer = Likes.beer))",

        # Same table in outer and subquery (conflict case)
        "SELECT * FROM Frequents WHERE EXISTS (SELECT * FROM Frequents WHERE Frequents.drinker = 'Bob')",

        # Same table with alias in outer query
        "SELECT * FROM Frequents f1 WHERE EXISTS (SELECT * FROM Frequents WHERE Frequents.drinker = f1.drinker)",

        # Multiple EXISTS with same table
        "SELECT * FROM Frequents WHERE EXISTS (SELECT * FROM Likes WHERE Likes.drinker = Frequents.drinker) AND EXISTS (SELECT * FROM Likes WHERE Likes.beer = 'Bud')",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Original: {q}")
        try:
            rewriter = ExistsRewriter(q)
            has_exists = rewriter.contains_exists()
            print(f"Has EXISTS: {has_exists}")

            if has_exists:
                rewritten = rewriter.rewrite()
                print(f"Rewritten: {rewritten}")
        except Exception as e:
            import traceback
            print(f"Error: {e}")
            traceback.print_exc()
