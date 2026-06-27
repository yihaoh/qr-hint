# built-in packages
from copy import deepcopy

# extra packages
from z3 import *
from .query_info import *

# project packages
from .boolean_parse_tree import BNode
from .utils import *
from .subtree_iter import *
from .fix_generator import FixGenerator
from .fix_optimizer import FixOptimizer
from .exists_rewriter import ExistsRewriter


def preprocess_query_with_exists(q2: str) -> tuple[str, dict]:
    """
    Preprocess Q2 to rewrite EXISTS subqueries before creating QueryInfo.

    This function should be called BEFORE creating MappingInfo, so that
    table mapping can correctly handle the rewritten query.

    Args:
        q2: Original Q2 query string

    Returns:
        tuple: (rewritten_query, rewrite_info_dict)
            - rewritten_query: The rewritten SQL (same as input if no EXISTS)
            - rewrite_info_dict: Dictionary with rewrite information

    Raises:
        RuntimeError: If Q2 contains NOT EXISTS or rewrite fails

    Example:
        >>> q2 = "SELECT * FROM T WHERE EXISTS (SELECT * FROM S WHERE S.id = T.id)"
        >>> rewritten_q2, info = preprocess_query_with_exists(q2)
        >>> # Now create QueryInfo and MappingInfo with rewritten query
        >>> q2_info = QueryInfo(rewritten_q2)
    """
    rewrite_info = {
        "has_exists": False,
        "rewrite_attempted": False,
        "rewrite_successful": False,
        "original_q2": q2,
        "rewritten_q2": None,
        "error": None,
        "error_type": None
    }

    try:
        # Create rewriter to check for EXISTS
        rewriter = ExistsRewriter(q2)

        # Check if Q2 has EXISTS or NOT EXISTS
        has_exists = rewriter.contains_exists()

        if not has_exists:
            # No EXISTS, return original query
            rewrite_info["rewritten_q2"] = q2
            return q2, rewrite_info

        # Mark that Q2 has EXISTS
        rewrite_info["has_exists"] = True

        # Check for NOT EXISTS (unsupported)
        def _contains_not_exists(xnode) -> bool:
            if isinstance(xnode, dict):
                if xnode.get("type") == "XBasicCallNode":
                    if xnode.get("operator_name", "") == "NOT EXISTS":
                        return True
                for key, value in xnode.items():
                    if isinstance(value, (dict, list)):
                        if _contains_not_exists(value):
                            return True
            elif isinstance(xnode, list):
                for item in xnode:
                    if _contains_not_exists(item):
                        return True
            return False

        if _contains_not_exists(rewriter.xtree):
            rewrite_info["rewrite_attempted"] = True
            rewrite_info["rewrite_successful"] = False
            rewrite_info["error_type"] = "NOT_EXISTS_UNSUPPORTED"
            rewrite_info["error"] = "NOT EXISTS subqueries are not supported. Cannot rewrite query with NOT EXISTS."
            raise RuntimeError(rewrite_info["error"])

        # Try to rewrite
        rewrite_info["rewrite_attempted"] = True
        rewritten_query = rewriter.rewrite()

        # Rewrite successful
        rewrite_info["rewrite_successful"] = True
        rewrite_info["rewritten_q2"] = rewritten_query

        return rewritten_query, rewrite_info

    except RuntimeError:
        # Re-raise RuntimeError (NOT EXISTS or other known issues)
        raise
    except Exception as e:
        # Unexpected error during rewrite
        rewrite_info["rewrite_attempted"] = True
        rewrite_info["rewrite_successful"] = False
        rewrite_info["error_type"] = "UNEXPECTED_ERROR"
        rewrite_info["error"] = f"Unexpected error during EXISTS rewrite: {str(e)}"
        raise RuntimeError(rewrite_info["error"]) from e


class QueryTest:
    """
    Class that contains following test:
    WHERE, GROUP BY, HAVING, SELECT

    Attributes
    ----------
    schema: dict
        table name --> [[type], [attr]]
    q1_xtree: dict
        q1 xtree
    q2_xtree: dict
        q2 xtree
    z3_var:
        table alias --> [z3 var instance, one for each attr in table]
    mapping: list
        a pair of dict, table alias --> [mutual alias]
    solver: Solver
        z3 solver object
    q1_where_tree: BNode
        syntax tree of q1 WHERE clause
    q2_where_tree BNode
        syntax tree of q2 WHERE clause


    Setup/Util Methods
    ------------------
    build_syntax_tree()
    trace_std_alias()
    convert_to_nary_tree()
    check_implication()
    create_bounds()
    create_bounds_nary()
    verify_repair_sites()
    find_smallest_rs()

    Test Methods
    ------------

    """

    def __init__(
        self, q1_info: QueryInfo, q2_info: QueryInfo, z3_lookup: dict, mapping, reverse_mapping, schema=db_schema
    ):
        self.schema = schema
        self.q1_info: QueryInfo = q1_info
        self.q2_info: QueryInfo = q2_info

        self.z3_var = z3_lookup
        self.z3_var_g = {}
        self.mapping = mapping
        self.reverse_mapping = reverse_mapping
        self.solver = Solver()

        self.q1_where_tree = self.build_syntax_tree(
            self.q1_info.flatten_where_trees, self.mapping[0], self.q1_info.attr_trace, "self.z3_var"
        )
        self.q2_where_tree = self.build_syntax_tree(
            self.q2_info.flatten_where_trees, self.mapping[1], self.q2_info.attr_trace, "self.z3_var"
        )

        self.q1_groupby_expr = (
            [
                self.build_syntax_tree(x, self.mapping[0], self.q1_info.attr_trace, "self.z3_var")
                for x in self.q1_info.flatten_groupby_exprs
            ]
            if q1_info.flatten_groupby_exprs
            else []
        )
        self.q2_groupby_expr = (
            [
                self.build_syntax_tree(x, self.mapping[1], self.q2_info.attr_trace, "self.z3_var")
                for x in self.q2_info.flatten_groupby_exprs
            ]
            if q2_info.flatten_groupby_exprs
            else []
        )

        self.q1_having_tree = self.build_syntax_tree(
            self.q1_info.flatten_having, self.mapping[0], self.q1_info.attr_trace, "self.z3_var"
        )
        self.q2_having_tree = self.build_syntax_tree(
            self.q2_info.flatten_having, self.mapping[1], self.q2_info.attr_trace, "self.z3_var"
        )

        self.q1_select_expr = (
            [
                self.build_syntax_tree(x, self.mapping[0], self.q1_info.attr_trace, "self.z3_var")
                for x in self.q1_info.flatten_select
            ]
            if self.q1_info.flatten_select
            else []
        )
        self.q2_select_expr = (
            [
                self.build_syntax_tree(x, self.mapping[1], self.q2_info.attr_trace, "self.z3_var")
                for x in self.q2_info.flatten_select
            ]
            if self.q2_info.flatten_select
            else []
        )

        # make sure both trees are n-ary
        self.q1_where_tree = convert_to_nary_tree(self.q1_where_tree)
        self.q2_where_tree = convert_to_nary_tree(self.q2_where_tree)
        self.q1_having_tree = convert_to_nary_tree(self.q1_having_tree)
        self.q2_having_tree = convert_to_nary_tree(self.q2_having_tree)

        # Test results are saved here
        self.fg = None
        self.fo = None
        self.rs_fix_pair_fg = []
        self.rs_fix_pair_fo = []
        self.where_res: tuple[list[str], list[str]] = None  # list repair sites, list of fixes
        self.group_by_res: tuple[list[str], list[str]] = None  # list of incorrect expr, list of missing expr
        self.having_res: tuple[list[str], list[str]] = None  # list repair sites, list of fixes
        self.select_res: tuple[list[str], list[str]] = None  # list of incorrect expr, list of missing expr

        # Store DISTINCT mismatch information
        self.distinct_mismatches: dict[str, list] = {
            "select": [],
            "having": []
        }

        # Store EXISTS rewrite information (optional, can be set externally)
        self.exists_rewrite_info: dict = None

        # =========================================================
        # WHERE/HAVING equivalence handling:
        #
        # Problem: Non-aggregate conditions can be placed in either WHERE or HAVING
        # and they are semantically equivalent. Examples:
        #
        # Correct: WHERE a=1 HAVING drinker='Amy' AND COUNT(*)>5
        # User v1: WHERE a=1 AND drinker='Amy' HAVING COUNT(*)>5     -> Should pass
        # User v2: WHERE a=1 HAVING drinker='Amy' AND COUNT(*)>5     -> Should pass
        # User v3: WHERE drinker='Amy' HAVING a=1 AND COUNT(*)>5     -> Should pass
        #
        # Solution:
        # - WHERE stage: Compare combined trees (WHERE + non-agg HAVING) from BOTH queries
        #   q1_where_combined = q1.WHERE + q1.non_agg_HAVING
        #   q2_where_combined = q2.WHERE + q2.non_agg_HAVING
        #   This catches all non-aggregate conditions regardless of WHERE/HAVING placement
        #
        # - HAVING stage: Compare ONLY aggregate conditions
        #   q1_having_agg_only = q1.HAVING conditions that contain aggregates
        #   q2_having_agg_only = q2.HAVING conditions that contain aggregates
        # =========================================================
        self.q1_having_non_agg = self._extract_non_aggregate_conditions(self.q1_having_tree)
        self.q2_having_non_agg = self._extract_non_aggregate_conditions(self.q2_having_tree)

        # Combined WHERE trees: WHERE + non-agg HAVING (for WHERE stage)
        self.q1_where_combined = self._combine_trees_with_and(self.q1_where_tree, self.q1_having_non_agg)
        self.q2_where_combined = self._combine_trees_with_and(self.q2_where_tree, self.q2_having_non_agg)

        # Aggregate-only HAVING trees (for HAVING stage)
        self.q1_having_agg_only = self._keep_only_aggregate_conditions(self.q1_having_tree)
        self.q2_having_agg_only = self._keep_only_aggregate_conditions(self.q2_having_tree)

    # =============================================================
    # Setup/util functions section
    # =============================================================

    def _contains_aggregate(self, node: BNode) -> bool:
        """Check if BNode tree contains aggregate function."""
        if not node:
            return False
        if node.type == "expr" and node.val in ["COUNT", "COUNTINT", "COUNTSTR", "SUM", "AVG", "MAX", "MIN"]:
            return True
        for child in node.children:
            if self._contains_aggregate(child):
                return True
        return False

    def _flatten_and_tree(self, node: BNode) -> list:
        """Flatten AND tree into list of conjuncts."""
        if not node:
            return []
        if node.type == "log" and node.val == "And":
            result = []
            for child in node.children:
                result.extend(self._flatten_and_tree(child))
            return result
        return [node]

    def _build_and_tree(self, conditions: list) -> BNode:
        """Build AND tree from list of conditions."""
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        result = BNode("And", "log")
        result.children = [conditions[0], conditions[1]]
        for i in range(2, len(conditions)):
            new_result = BNode("And", "log")
            new_result.children = [result, conditions[i]]
            result = new_result
        return result

    def _extract_non_aggregate_conditions(self, having_tree: BNode) -> list:
        """Extract non-aggregate conditions from HAVING tree as list."""
        if not having_tree:
            return []
        conjuncts = self._flatten_and_tree(having_tree)
        return [c for c in conjuncts if not self._contains_aggregate(c)]

    def _keep_only_aggregate_conditions(self, having_tree: BNode) -> BNode:
        """Keep only aggregate conditions in HAVING, return new tree."""
        if not having_tree:
            return None
        conjuncts = self._flatten_and_tree(having_tree)
        agg_conjuncts = [c for c in conjuncts if self._contains_aggregate(c)]
        return self._build_and_tree(agg_conjuncts)

    def _combine_trees_with_and(self, tree: BNode, conditions: list) -> BNode:
        """Combine tree with list of conditions using AND."""
        if not conditions:
            return tree
        if not tree:
            return self._build_and_tree(conditions)
        all_conditions = self._flatten_and_tree(tree) + conditions
        return self._build_and_tree(all_conditions)

    def _clone_bnode_with_g_vars(self, node: BNode) -> BNode:
        """
        Clone a BNode tree, replacing z3_var references with z3_var_g references.
        This is needed for GROUP BY checking where we need .g variable versions.
        """
        if not node:
            return None

        # Create new node with same properties
        new_node = BNode(node.val, node.type, node.select_xid, None, node.is_distinct)

        # If it's a variable reference, convert to .g version
        if node.type == "var" and "self.z3_var[" in node.val:
            # Convert self.z3_var["X"][i] to self.z3_var_g["X"][i]
            new_node.val = node.val.replace("self.z3_var[", "self.z3_var_g[")

        # Recursively clone children
        for child in node.children:
            cloned_child = self._clone_bnode_with_g_vars(child)
            if cloned_child:
                cloned_child.parent = new_node
                new_node.children.append(cloned_child)

        return new_node

    def build_syntax_tree(
        self,
        xnode,
        alias_to_mutual: dict[str, str],
        attr_trace: dict[tuple[str, str, str], tuple[str, str]], # NOTE: second tuple should be 3-tuple?
        z3_dict: str,
        parent=None,
    ) -> BNode:
        """
        ### Input
            xnode: dict (json), xNode represent the subtree from Calcite
            alias_to_mutual: dict, table alias in cur query -> mutual alias
            attr_trace: dict, (outer xSelect_id, outer table, col) -> (inner xSelect_id, inner table, col)
            z3_dict: string, var name of the z3 var lookup table for all attributes, usually "self.z3_var"
            parent: BNode, the parent node of current node
        ### Return
            BNode, the built syntax tree with BNode
        ### Raises
            NotImplementedError: do not support certain xNode
        """
        if not xnode:
            return None

        select_xid = xnode["xid"] if "xid" in xnode else None

        if xnode["type"] == "XBasicCallNode" or xnode["type"] == "XConnector":
            if xnode["operator_name"] in log_ops or xnode["operator_name"] in cmp_ops:
                op_type = "log" if xnode["operator_name"].upper() in log_ops else "pred"
                n = BNode(xnode["operator_name"].capitalize(), op_type, select_xid, parent)
                if xnode["operator_name"].upper() == "NOT":
                    n.children.append(
                        self.build_syntax_tree(xnode["operands"][0], alias_to_mutual, attr_trace, z3_dict, n)
                    )
                else:
                    n.children.append(
                        self.build_syntax_tree(xnode["operands"][0], alias_to_mutual, attr_trace, z3_dict, n)
                    )
                    n.children.append(
                        self.build_syntax_tree(xnode["operands"][1], alias_to_mutual, attr_trace, z3_dict, n)
                    )
                return n

            elif xnode["operator_name"] in ari_ops:
                op_type = "expr"
                op_val = "+" if xnode["operator_name"] == "||" else xnode["operator_name"]
                n = BNode(op_val, op_type, select_xid, parent)
                n.children.append(self.build_syntax_tree(xnode["operands"][0], alias_to_mutual, attr_trace, z3_dict, n))
                n.children.append(self.build_syntax_tree(xnode["operands"][1], alias_to_mutual, attr_trace, z3_dict, n))
                return n

            elif xnode["operator_name"] in agg_ops:
                op_type, op_val = "expr", xnode["operator_name"]
                if xnode["operator_name"] == "COUNT":
                    op_val = "COUNTINT" if xnode["operands"][0]["data_type"] == "INTEGER" else "COUNTSTR"

                # Detect DISTINCT in aggregate function
                sql_string = xnode.get("sql_string", "")
                has_distinct = "DISTINCT" in sql_string.upper()

                n = BNode(op_val, op_type, select_xid, parent, is_distinct=has_distinct)
                n.children.append(self.build_syntax_tree(xnode["operands"][0], alias_to_mutual, attr_trace, z3_dict, n))
                return n

            else:
                # don't handle this operator
                raise NotImplementedError(
                    f'Error building syntax tree: Operator < {xnode["operator_name"]} > is not currently supported.'
                )

        elif xnode["type"] == "XColumnRefNode":
            # Double check to make sure form valid z3 formula
            attr_split = xnode["sql_string"].split(".")
            table_alias, column_alias = self.trace_std_alias(
                xnode["XSelectNode_id"], attr_split[0], attr_split[1], attr_trace
            )
            table_alias = alias_to_mutual[table_alias]
            table = table_alias.split("_")[0]
            col_idx = self.schema[table][1].index(column_alias)
            return BNode(f'{z3_dict}["{table_alias}"][{col_idx}]', "var", select_xid, parent)

        elif xnode["type"] == "XLiteralNode":
            literal_val = xnode["sql_string"]
            # Handle DATE literals: DATE 'YYYY-MM-DD' → YYYYMMDD integer for Z3 comparison
            if literal_val.upper().startswith("DATE '") and literal_val.endswith("'"):
                return BNode(date_literal_to_int(literal_val), "const", select_xid, parent)
            return BNode(literal_val, "const", select_xid, parent)

        elif xnode["type"] == "XColumnRenameNode":
            return self.build_syntax_tree(xnode["operand"], alias_to_mutual, attr_trace, z3_dict, parent)

        # Node type not supported
        raise NotImplementedError(f'Building Syntax Tree Error: Do not support node type {xnode["type"]}.')

    def trace_std_alias(self, select_xid: str, table_alias: str, col: str, attr_trace) -> tuple[str, str]:
        """Given select_xid, table_alias and column name, find its correspondance in inner-most context.
        ### Input
            select_xid: string, id of the root node of a SELECT context
            table_alias: string, table alias in the SELECT context
            col: string, the attribute of the table in the SELECT context
            attr_trace: dict, attr from outer context --> same attr from inner context

        ### Return
            (string, string), table name in the inner-most context and its attribute
        """
        next_attr = attr_trace[(select_xid, table_alias, col)]

        while next_attr[-1] != "std":
            next_attr = attr_trace[(next_attr[0], next_attr[1], next_attr[2])]
            if next_attr[-1] == "expr":
                raise ValueError(
                    f"bool_test: trace_std_alias: column ({next_attr[0], next_attr[1]}) did not expand properly."
                )

        return next_attr[0], next_attr[1] # t_0 , bar e.g.

    def check_implication(self, q1: str, q2: str, contexts: list[str] = []) -> bool:
        """Test if q1 -> q2 holds true
        ### Input
            q1: string, a formula ready for eval()
            q2: string, a formula ready for eval()

        ### Return
            bool, true of q1 -> q2, else false
        """
        self.solver.reset()
        for c in contexts:
            self.solver.add(eval(c))

        q1 = eval(q1) if type(q1) == str else q1
        q2 = eval(q2) if type(q2) == str else q2
        self.solver.add(Not(Implies(q1, q2)))
        return self.solver.check() == unsat

    def create_bounds_nary(self, syn_tree: BNode, disjoint_trees: list[BNode]) -> tuple[str, str]:
        """Create repair bounds at each node in the n-ary syntax tree.
        Return the repair bounds for entire syntax tree.

        ### Input
            syn_tree: BNode, root of the syntax tree
            disjoint_tree: BNode[], a set of repair sites

        ### Return
            (lowerbound, upperbound): both in z3-ready format
        """
        if not disjoint_trees:
            return (syn_tree.getZ3(), syn_tree.getZ3())

        if syn_tree in disjoint_trees:
            syn_tree.bounds = ("False", "True")
            return syn_tree.bounds
        elif not syn_tree.children or syn_tree.type == "pred":
            syn_tree.bounds = (syn_tree.getZ3(), syn_tree.getZ3())
            return syn_tree.bounds

        final_bounds = None

        if len(syn_tree.children) == 1:
            child_bounds = self.create_bounds_nary(syn_tree.children[0], disjoint_trees)
            final_bounds = [f"{syn_tree.val}({child_bounds[1]})", f"{syn_tree.val}({child_bounds[0]})"]
        elif len(syn_tree.children) > 1:
            child0_bounds = self.create_bounds_nary(syn_tree.children[0], disjoint_trees)
            child1_bounds = self.create_bounds_nary(syn_tree.children[1], disjoint_trees)
            final_bounds = [
                f"{syn_tree.val}({child0_bounds[0]}, {child1_bounds[0]})",
                f"{syn_tree.val}({child0_bounds[1]}, {child1_bounds[1]})",
            ]
            for i in range(2, len(syn_tree.children)):
                tmp = self.create_bounds_nary(syn_tree.children[i], disjoint_trees)
                final_bounds[0] = f"{syn_tree.val}({final_bounds[0]}, {tmp[0]})"
                final_bounds[1] = f"{syn_tree.val}({final_bounds[1]}, {tmp[1]})"

        syn_tree.bounds = tuple(final_bounds)
        return syn_tree.bounds

    def verify_repair_sites(self, q1: BNode, q2: BNode, disjoint_trees: list[BNode]) -> bool:
        """Verify if a set of disjoint trees is a set of repair sites.
        ### Input
            q1: target (correct) formula
            q2: wrong formula
            disjoint_trees: BNode[], a set of distjoint trees
        ### Return
            bool, true if valid repair sites, else false
        """
        lower, upper = self.create_bounds_nary(q2, disjoint_trees)
        target_formula = q1.getZ3()

        if self.check_implication(lower, target_formula) and self.check_implication(target_formula, upper):
            return True

        return False

    # =============================================================
    # Test fucntions section.
    # =============================================================
    def test_where_fo(self, num_rs=2):
        # make sure both queries have WHERE
        if not self.q1_where_tree and not self.q2_where_tree:
            print("WHERE/HAVING clauses are equivalent!")
            return
        elif not self.q1_where_tree:
            print("Do not need WHERE/HAVING")
            self.rs_fix_pair_fo = [
                (
                    (
                        translate_query_namespace(
                            eval(self.q2_where_tree.getZ3()), self.reverse_mapping[1], self.q2_info.std_alias_to_info
                        ),
                        None,
                    ),
                    (self.q2_where_tree.get_size(), 0),
                )
            ]
            return
        elif not self.q2_where_tree:
            print("Missing WHERE/HAVING")
            self.rs_fix_pair_fo = [
                (
                    (
                        None,
                        translate_query_namespace(
                            eval(self.q1_where_tree.getZ3()), self.reverse_mapping[1], self.q2_info.std_alias_to_info
                        ),
                    ),
                    (0, self.q1_where_tree.get_size()),
                )
            ]
            return
        elif self.check_implication(self.q1_where_tree.getZ3(), self.q2_where_tree.getZ3()) and self.check_implication(
            self.q2_where_tree.getZ3(), self.q1_where_tree.getZ3()
        ):
            # Check where equivalence
            print("WHERE/HAVING clauses are equivalent!")
            return

        final_fixes = None
        cost = None

        iter = RepairSitesIter(self.q2_where_tree, num_rs, self.q1_where_tree.get_size(), self.q2_where_tree.get_size())
        cur_rs = iter.next()
        is_conj = is_conjunctive(self.q2_where_tree)

        while cur_rs:
            # print('cur_rs len: ', len(cur_rs))
            repair_site_sz = sum([s.get_size() for s in cur_rs]) / (
                self.q1_where_tree.get_size() + self.q2_where_tree.get_size()
            ) + 1 / 6 * get_actual_rs_count(cur_rs)
            if cost is not None and repair_site_sz > cost:
                break

            if not self.verify_repair_sites(self.q1_where_tree, self.q2_where_tree, cur_rs):
                cur_rs = iter.next()
                continue

            target_rs, lower, upper = lca_nary(
                self.q2_where_tree, self.q1_where_tree.getZ3(), self.q1_where_tree.getZ3()
            )
            # print(eval(target_rs.getZ3()))
            fo = FixOptimizer(eval(lower), eval(upper), target_rs, cur_rs, self.z3_var)
            cur_cost = repair_cost(self.q1_where_tree, self.q2_where_tree, fo.get_fixes())
            if cost == None or cur_cost < cost:
                final_fixes = fo.get_fixes()
                cost = cur_cost
                self.fo = fo
            cur_rs = iter.next()

            if is_conj:
                break

        self.fo_min_cost = cost
        for s in final_fixes:
            tmp_rs = [
                translate_query_namespace(eval(x.getZ3()), self.reverse_mapping[1], self.q2_info.std_alias_to_info)
                for x in s[0]
            ]
            tmp_rs_sz = sum([x.get_size() for x in s[0]])
            tmp_f = translate_query_namespace(s[1][0], self.reverse_mapping[1], self.q2_info.std_alias_to_info)
            tmp_f_sz = s[1][1]

            self.rs_fix_pair_fo.append(((tmp_rs, tmp_f), (tmp_rs_sz, tmp_f_sz)))
        self.print_rs_fixes(self.rs_fix_pair_fo)

    def test_where_fg(self, num_rs=2):
        # Use combined WHERE trees that include non-aggregate HAVING conditions
        # This allows conditions to be placed in either WHERE or HAVING
        q1_combined = self.q1_where_combined
        q2_combined = self.q2_where_combined

        # make sure both queries have WHERE (or non-agg HAVING conditions)
        if not q1_combined and not q2_combined:
            print("WHERE clauses (+ non-agg HAVING) are equivalent!")
            return
        elif not q1_combined:
            print("Do not need WHERE clause (or non-agg HAVING)")
            self.rs_fix_pair_fg.append(
                (
                    (
                        translate_query_namespace(
                            eval(q2_combined.getZ3()), self.reverse_mapping[1], self.q2_info.std_alias_to_info
                        ),
                        None,
                    ),
                    (q2_combined.get_size(), 0),
                )
            )
            return
        elif not q2_combined:
            print("Missing WHERE clause (or non-agg HAVING)")
            self.rs_fix_pair_fg.append(
                (
                    (
                        None,
                        translate_query_namespace(
                            eval(q1_combined.getZ3()), self.reverse_mapping[1], self.q2_info.std_alias_to_info
                        ),
                    ),
                    (0, q1_combined.get_size()),
                )
            )
            return
        elif self.check_implication(q1_combined.getZ3(), q2_combined.getZ3()) and self.check_implication(
            q2_combined.getZ3(), q1_combined.getZ3()
        ):
            # Check where equivalence (including non-agg HAVING conditions)
            print("WHERE clauses (+ non-agg HAVING) are equivalent!")
            return

        # Store top 3 solutions: list of (cost, fixes, fg) tuples
        top_solutions = []

        # Use combined trees for repair iteration (WHERE + non-agg HAVING)
        # Make combined trees n-ary for repair site iteration
        q1_combined_nary = convert_to_nary_tree(q1_combined)
        q2_combined_nary = convert_to_nary_tree(q2_combined)

        iter = RepairSitesIter(q2_combined_nary, num_rs, q1_combined_nary.get_size(), q2_combined_nary.get_size())
        cur_rs = iter.next()
        is_conj = is_conjunctive(q2_combined_nary)

        while cur_rs:
            repair_site_sz = sum([s.get_size() for s in cur_rs]) / (
                q1_combined_nary.get_size() + q2_combined_nary.get_size()
            ) + 1 / 6 * get_actual_rs_count(cur_rs)
            # Check against max cost in top_solutions if we have 3 solutions
            if len(top_solutions) >= 3 and repair_site_sz > top_solutions[-1][0]:
                break

            if not self.verify_repair_sites(q1_combined_nary, q2_combined_nary, cur_rs):
                cur_rs = iter.next()
                continue

            fg = FixGenerator(q1_combined_nary, q2_combined_nary, self.z3_var, cur_rs)
            res = fg.get_fixes()

            cur_cost = repair_cost(q1_combined_nary, q2_combined_nary, res)

            # Insert solution into top_solutions maintaining cost order
            top_solutions.append((cur_cost, res, fg))
            top_solutions.sort(key=lambda x: x[0])  # Sort by cost (smallest first)

            # Keep only top 3 solutions
            if len(top_solutions) > 3:
                top_solutions = top_solutions[:3]

            cur_rs = iter.next()

            if is_conj:
                print("query is conjunctive")
                break

        # Store minimum cost and first fix generator
        if len(top_solutions) > 0:
            self.fg_min_cost = top_solutions[0][0]
            self.fg = top_solutions[0][2]
        else:
            self.fg_min_cost = None

        # Process all solutions in top_solutions
        for solution_cost, final_fixes, fg_obj in top_solutions:
            solution_repairs = []
            for s in final_fixes:
                tmp_rs = [
                    translate_query_namespace(eval(x.getZ3()), self.reverse_mapping[1], self.q2_info.std_alias_to_info)
                    for x in s[0]
                ]
                tmp_rs_sz = sum([x.get_size() for x in s[0]])
                tmp_f = translate_query_namespace(s[1][0], self.reverse_mapping[1], self.q2_info.std_alias_to_info)
                tmp_f_sz = s[1][1]

                solution_repairs.append(((tmp_rs, tmp_f), (tmp_rs_sz, tmp_f_sz)))

            # Store each solution's repairs with cost
            self.rs_fix_pair_fg.append({
                'cost': solution_cost,
                'repairs': solution_repairs
            })

        self.print_rs_fixes(self.rs_fix_pair_fg)

    def print_rs_fixes(self, rs_fix_pair):
        for solution_idx, solution in enumerate(rs_fix_pair):
            # Handle new dictionary format
            if isinstance(solution, dict):
                print(f"\n=== Solution #{solution_idx + 1} (Cost: {solution['cost']}) ===")
                for i, (rs_f, sizes) in enumerate(solution['repairs']):
                    print(f"Repair Site #{i}: {rs_f[0]}")
                    print(f"Fix #{i}: {rs_f[1]}")
            else:
                # Handle old format for backward compatibility
                rs_f, sizes = solution
                print(f"Repair Site #{solution_idx}: {rs_f[0]}")
                print(f"Fix #{solution_idx}: {rs_f[1]}")

    def test_group_by(self):
        # prepare for group by check
        if not self.q1_groupby_expr and self.q2_groupby_expr:
            # print('Should not have any group by expressions.')
            self.group_by_res = ["Should not use GROUP BY clause."], []
            return
        elif self.q1_groupby_expr and not self.q2_groupby_expr:
            # print('Need to use group by in your query.')
            self.group_by_res = [], ["Missing GROUP BY clause."]
            return
        elif not self.q1_groupby_expr and not self.q2_groupby_expr:
            # print('Both queries have no group by')
            self.group_by_res = [], []
            return

        # print("q1_groupby_expr: ", self.q1_groupby_expr) # test
        # print("q2_groupby_expr: ", self.q2_groupby_expr) # test

        # build z3_var_g to check group by
        for key, value in self.z3_var.items():
            self.z3_var_g[key] = []
            for v in value:
                ty = str(v.sort())
                if ty == "Int":
                    self.z3_var_g[key].append(Int(f"{str(v)}.g"))
                elif ty == "String":
                    self.z3_var_g[key].append(String(f"{str(v)}.g"))
                elif ty == "Real":
                    self.z3_var_g[key].append(Real(f"{str(v)}.g"))
        print("z3_var: ", self.z3_var) # test: print z3_var
        print("z3_var_g: ", self.z3_var_g) # test

        # =========================================================
        # GROUP BY equivalence with WHERE/HAVING flexibility:
        #
        # When a column is fixed to a constant in WHERE (e.g., drinker='Amy'),
        # it doesn't need to be in GROUP BY. But if the same condition is in HAVING,
        # the column MUST be in GROUP BY.
        #
        # Solution: Use combined WHERE trees (WHERE + non-agg HAVING) for each query
        # to determine what conditions are effectively applied before GROUP BY.
        # This allows user's GROUP BY to omit columns that are constants in their WHERE.
        # =========================================================

        # Build WHERE expressions for GROUP BY check using combined trees
        # q1_where_combined and q2_where_combined already include non-agg HAVING conditions
        q1where_g = None
        q2where_g = None

        # Clone q1's combined WHERE for GROUP BY check
        if self.q1_where_combined:
            # We need to rebuild from flatten_where_trees + non-agg having for .g variables
            q1where_g = self.build_syntax_tree(
                self.q1_info.flatten_where_trees, self.mapping[0], self.q1_info.attr_trace, "self.z3_var_g"
            )
            # Also add non-agg HAVING conditions with .g variables
            for cond in self.q1_having_non_agg:
                if q1where_g:
                    new_node = BNode("And", "log")
                    # Clone the condition with .g variables - we need to rebuild it
                    new_node.children = [q1where_g, self._clone_bnode_with_g_vars(cond)]
                    q1where_g = new_node
                else:
                    q1where_g = self._clone_bnode_with_g_vars(cond)

        # Clone q2's combined WHERE for GROUP BY check
        if self.q2_where_combined:
            q2where_g = self.build_syntax_tree(
                self.q2_info.flatten_where_trees, self.mapping[1], self.q2_info.attr_trace, "self.z3_var_g"
            )
            # Also add non-agg HAVING conditions with .g variables
            for cond in self.q2_having_non_agg:
                if q2where_g:
                    new_node = BNode("And", "log")
                    new_node.children = [q2where_g, self._clone_bnode_with_g_vars(cond)]
                    q2where_g = new_node
                else:
                    q2where_g = self._clone_bnode_with_g_vars(cond)

        print("q1where_g: ", q1where_g) # test

        # combine two boolean formulae - use each query's OWN combined WHERE
        if self.q1_where_combined:
            q1exp = BNode("And", "log")
            q1exp.children = [self.q1_where_combined, q1where_g]
            print("q1exp: ", q1exp) # test
        else:
            q1exp = None

        # For q2, use q2's combined WHERE (not q1's!)
        if self.q2_where_combined:
            q2exp = BNode("And", "log")
            q2exp.children = [self.q2_where_combined, q2where_g]
            print("q2exp: ", q2exp) # test
        else:
            q2exp = None

        # clone the group by expression
        q1g = [
            self.build_syntax_tree(x, self.mapping[0], self.q1_info.attr_trace, "self.z3_var_g")
            for x in self.q1_info.flatten_groupby_exprs
        ]
        q2g = [
            self.build_syntax_tree(x, self.mapping[1], self.q2_info.attr_trace, "self.z3_var_g")
            for x in self.q2_info.flatten_groupby_exprs
        ]
        print("q1g: ", q1g) # test: print groupby columns
        print("q2g: ", q2g) # test: print groupby columns

        # construct group by predicates     NOTE: quesiton: what does it do? 
        q1gexp, q2gexp = [], []
        for i in range(len(q1g)):
            n = BNode("=", "pred")
            n.children = [self.q1_groupby_expr[i], q1g[i]]
            q1gexp.append(n)
        for i in range(len(q2g)):
            n = BNode("=", "pred")
            n.children = [self.q2_groupby_expr[i], q2g[i]]
            q2gexp.append(n)
        print("q1gexp: ", q1gexp) # test: print groupby predicates
        print("q2gexp: ", q2gexp) # test: print groupby predicates

        # parse group by into trees
        q1gt, q2gt = None, None
        if len(q1gexp) == 1:
            q1gt = q1gexp[0]
        else:
            q1gt = BNode("And", "log")
            q1gt.children = [q1gexp[0], q1gexp[1]]
            for i in range(2, len(q1gexp)):
                tp = q1gt
                q1gt = BNode("And", "log")
                q1gt.children = [tp, q1gexp[i]]

        if len(q2gexp) == 1:
            q2gt = q2gexp[0]
        else:
            q2gt = BNode("And", "log")
            q2gt.children = [q2gexp[0], q2gexp[1]]
            for i in range(2, len(q2gexp)):
                tp = q2gt
                q2gt = BNode("And", "log")
                q2gt.children = [tp, q2gexp[i]]

        print("q1gt: ", q1gt) # test: print groupby trees
        print("q2gt: ", q2gt) # test: print groupby trees

        # conjunct with boolean formula NOTE: no need to conjunct with boolean formula if q1exp is empty
        if q1exp:
            q1final = BNode("And", "log")
            q1final.children = [q1gt, q1exp]
            print("q1final: ", q1final) # test
        else:
            q1final = q1gt
        q1final_z3 = q1final.getZ3()

        # NOTE: no need to conjunct with boolean formula if q2exp is empty
        if q2exp:
            q2final = BNode("And", "log")
            q2final.children = [q2gt, q2exp]
            print("q2final: ", q2final) # test
        else:
            q2final = q2gt
        q2final_z3 = q2final.getZ3()

        print("q1final_z3: ", q1final_z3) # test
        print("q2final_z3: ", q2final_z3) # test

        # make sure they are not equivalent
        if self.check_implication(q1final_z3, q2final_z3) and self.check_implication(q2final_z3, q1final_z3):
            print("GROUP BY clauses are equivalent.")
            self.group_by_res = [], []
            return

        missing: list[str] = []
        incorrect: list[str] = []
        for i in range(len(q2gexp)):
            tp = BNode("And", "log")
            tp.children = [q2exp, q2gexp[i]]
            if not self.check_implication(q1final_z3, tp.getZ3()):
                incorrect.append(
                    translate_query_namespace(
                        eval(self.q2_groupby_expr[i].getZ3()), self.reverse_mapping[1], self.q2_info.std_alias_to_info
                    )
                )

        for i in range(len(q1gexp)):
            tp = BNode("And", "log")
            tp.children = [q1exp, q1gexp[i]]
            if not self.check_implication(q2final_z3, tp.getZ3()):
                missing.append(
                    translate_query_namespace(
                        eval(self.q1_groupby_expr[i].getZ3()), self.reverse_mapping[1], self.q2_info.std_alias_to_info
                    )
                )

        self.group_by_res = incorrect, missing
        return

    def test_having(self, num_rs=2):
        # TODO: finish having
        # make sure both queries have WHERE
        if not self.q1_having_tree and not self.q2_having_tree:
            print("WHERE/HAVING clauses are equivalent!")
            return
        elif not self.q1_having_tree:
            print("Do not need WHERE/HAVING")
            self.rs_fix_pair_fg.append(
                (
                    (
                        translate_query_namespace(
                            eval(self.q2_having_tree.getZ3()), self.reverse_mapping[1], self.q2_info.std_alias_to_info
                        ),
                        None,
                    ),
                    (self.q2_having_tree.get_size(), 0),
                )
            )
            return
        elif not self.q2_having_tree:
            print("Missing WHERE/HAVING")
            self.rs_fix_pair_fg.append(
                (
                    (
                        None,
                        translate_query_namespace(
                            eval(self.q1_having_tree.getZ3()), self.reverse_mapping[1], self.q2_info.std_alias_to_info
                        ),
                    ),
                    (0, self.q1_having_tree.get_size()),
                )
            )
            return
        elif self.check_implication(
            self.q1_having_tree.getZ3(), self.q2_having_tree.getZ3(), [self.q1_where_tree.getZ3()]
        ) and self.check_implication(
            self.q2_having_tree.getZ3(), self.q1_having_tree.getZ3(), [self.q1_where_tree.getZ3()]
        ):
            # Check where equivalence
            print("WHERE/HAVING clauses are equivalent!")
            return

        final_fixes = None
        cost = None

        iter = RepairSitesIter(
            self.q2_having_tree, num_rs, self.q1_having_tree.get_size(), self.q2_having_tree.get_size()
        )
        cur_rs: list[BNode] = iter.next()
        is_conj = is_conjunctive(self.q2_having_tree)

        while cur_rs:
            repair_site_sz = sum([s.get_size() for s in cur_rs]) / (
                self.q1_having_tree.get_size() + self.q2_having_tree.get_size()
            ) + 1 / 6 * get_actual_rs_count(cur_rs) # use len(cur_rs) instead of get_actual_rs_count(cur_rs)
            if cost is not None and repair_site_sz > cost:  # or repair_site_sz > self.q2_comb_tree.get_size() / 2:
                break

            if not self.verify_repair_sites(self.q1_having_tree, self.q2_having_tree, cur_rs):
                cur_rs = iter.next()
                continue

            fg = FixGenerator(self.q1_having_tree, self.q2_having_tree, self.z3_var, cur_rs)
            res = fg.get_fixes()

            cur_cost = repair_cost(self.q1_having_tree, self.q2_having_tree, res)

            if cost == None or cur_cost < cost:
                final_fixes = res
                cost = cur_cost
                self.fg = fg
            cur_rs = iter.next()

            if is_conj:
                print("query is conjunctive")
                break

        self.fg_min_cost = cost
        for s in final_fixes:
            tmp_rs = [
                translate_query_namespace(eval(x.getZ3()), self.reverse_mapping[1], self.q2_info.std_alias_to_info)
                for x in s[0]
            ]
            tmp_rs_sz = sum([x.get_size() for x in s[0]])
            tmp_f = translate_query_namespace(s[1][0], self.reverse_mapping[1], self.q2_info.std_alias_to_info)
            tmp_f_sz = s[1][1]

            self.rs_fix_pair_fg.append(((tmp_rs, tmp_f), (tmp_rs_sz, tmp_f_sz)))

        self.print_rs_fixes(self.rs_fix_pair_fg)

    def _find_aggregate_distinct_mismatches(self, node1: BNode, node2: BNode, mismatches: list = None) -> list:
        """
        Recursively find all aggregate functions where DISTINCT usage doesn't match.

        Args:
            node1: First expression tree node
            node2: Second expression tree node
            mismatches: List to accumulate mismatch information

        Returns:
            list: List of (node1_info, node2_info) tuples where DISTINCT doesn't match
                  Each info is: (aggregate_type, is_distinct, z3_formula)
        """
        if mismatches is None:
            mismatches = []

        if not node1 or not node2:
            return mismatches

        # Check if both nodes are aggregate functions
        if node1.type == "expr" and node1.val in ["COUNT", "COUNTINT", "COUNTSTR", "SUM", "AVG", "MAX", "MIN"]:
            if node2.type == "expr" and node2.val in ["COUNT", "COUNTINT", "COUNTSTR", "SUM", "AVG", "MAX", "MIN"]:
                # Both are aggregates - check if DISTINCT matches
                if node1.is_distinct != node2.is_distinct:
                    mismatch_info = {
                        "q1_aggregate": node1.val,
                        "q1_distinct": node1.is_distinct,
                        "q1_formula": node1.getZ3(),
                        "q2_aggregate": node2.val,
                        "q2_distinct": node2.is_distinct,
                        "q2_formula": node2.getZ3(),
                    }
                    mismatches.append(mismatch_info)

        # Recursively check children
        if len(node1.children) == len(node2.children):
            for c1, c2 in zip(node1.children, node2.children):
                self._find_aggregate_distinct_mismatches(c1, c2, mismatches)

        return mismatches

    def _check_aggregate_distinct_match(self, node1: BNode, node2: BNode) -> bool:
        """
        Check if two expression trees have matching DISTINCT modifiers
        on all aggregate functions.

        Returns:
            bool: True if DISTINCT usage matches, False otherwise
        """
        mismatches = self._find_aggregate_distinct_mismatches(node1, node2)
        return len(mismatches) == 0

    def _contains_aggregate_function(self, node: BNode) -> bool:
        """
        Check if an expression tree contains any aggregate function.

        Args:
            node: Expression tree node

        Returns:
            bool: True if tree contains SUM, AVG, COUNT, MAX, or MIN
        """
        if not node:
            return False

        # Check if current node is an aggregate function
        if node.type == "expr" and node.val in ["COUNT", "COUNTINT", "COUNTSTR", "SUM", "AVG", "MAX", "MIN"]:
            return True

        # Recursively check children
        for child in node.children:
            if self._contains_aggregate_function(child):
                return True

        return False

    def test_select(self):
        # Check for DISTINCT mismatches in aggregate functions within SELECT expressions
        # Store these separately so user can see both DISTINCT issues and semantic issues
        for i in range(len(self.q2_select_expr)):
            for j in range(len(self.q1_select_expr)):
                mismatches = self._find_aggregate_distinct_mismatches(
                    self.q1_select_expr[j],
                    self.q2_select_expr[i]
                )
                if mismatches:
                    for mismatch in mismatches:
                        mismatch["q1_expr_index"] = j
                        mismatch["q2_expr_index"] = i
                        self.distinct_mismatches["select"].append(mismatch)

        # Check query-level DISTINCT and store if different
        if self.q1_info.is_distinct != self.q2_info.is_distinct:
            # Store query-level DISTINCT mismatch
            query_level_mismatch = {
                "type": "query_level",
                "q1_distinct": self.q1_info.is_distinct,
                "q2_distinct": self.q2_info.is_distinct
            }
            self.distinct_mismatches["select"].append(query_level_mismatch)
            # Note: We continue checking semantic equivalence even if query-level DISTINCT differs

        # Continue with semantic equivalence checking
        select_err = []
        select_missing_idx = [True for _ in range(len(self.q1_select_expr))]
        select_out_of_place_idx = [True for _ in range(len(self.q2_select_expr))]

        # Helper function to check if two SELECT expressions are equivalent
        def check_select_equivalence(i, j):
            """Check if q2_select_expr[i] is equivalent to q1_select_expr[j]"""
            rhs = f"{self.q1_select_expr[j].getZ3()} == {self.q2_select_expr[i].getZ3()}"
            where_formula = self.q1_where_tree.getZ3() if self.q1_where_tree else "True"

            try:
                is_equivalent = False

                # First try standard implication check (with WHERE context)
                try:
                    is_equivalent = self.check_implication(eval(where_formula), eval(rhs))
                except:
                    # Standard check failed (e.g., SUM not defined in eval scope)
                    pass

                # If standard check fails, try with simplified aggregate axioms
                if not is_equivalent:
                    has_agg_q1 = self._contains_aggregate_function(self.q1_select_expr[j])
                    has_agg_q2 = self._contains_aggregate_function(self.q2_select_expr[i])

                    if has_agg_q1 or has_agg_q2:
                        q1_expr = self.q1_select_expr[j].getZ3()
                        q2_expr = self.q2_select_expr[i].getZ3()

                        try:
                            q1_str = str(q1_expr)
                            q2_str = str(q2_expr)
                            equality_formula = f"({q1_str}) == ({q2_str})"
                            forward = self.check_implication_with_simple_agg_axioms("True", equality_formula, [])
                            is_equivalent = forward
                        except Exception:
                            pass

                return is_equivalent
            except Exception:
                return False

        for i in range(len(self.q2_select_expr)):
            satisfied = False

            # Priority 1: Try matching with same position (i == j) first
            # This ensures that when columns are semantically equivalent due to WHERE conditions,
            # we prefer matching columns at the same position
            if i < len(self.q1_select_expr) and select_missing_idx[i]:
                if check_select_equivalence(i, i):
                    select_missing_idx[i] = False
                    select_out_of_place_idx[i] = False  # Same position, not out of place
                    satisfied = True

            # Priority 2: If same position doesn't match, try other positions
            if not satisfied:
                for j in range(len(self.q1_select_expr)):
                    if j == i:  # Skip already tried position
                        continue
                    if not select_missing_idx[j]:  # Skip already matched q1 expressions
                        continue

                    if check_select_equivalence(i, j):
                        select_missing_idx[j] = False
                        satisfied = True
                        # i != j, so select_out_of_place_idx[i] stays True (out of place)
                        break

            if not satisfied:
                select_out_of_place_idx[i] = False
                select_err.append(
                    translate_query_namespace(
                        eval(self.q2_select_expr[i].getZ3()), self.reverse_mapping[1], self.q2_info.std_alias_to_info
                    )
                )

        select_missing, select_out_of_place = [], []
        for i in range(len(self.q1_select_expr)):
            if select_missing_idx[i]:
                select_missing.append(
                    translate_query_namespace(
                        eval(self.q1_select_expr[i].getZ3()), self.reverse_mapping[0], self.q1_info.std_alias_to_info
                    )
                )

        for i in range(len(self.q2_select_expr)):
            if select_out_of_place_idx[i]:
                select_out_of_place.append(
                    translate_query_namespace(
                        eval(self.q2_select_expr[i].getZ3()), self.reverse_mapping[1], self.q2_info.std_alias_to_info
                    )
                )

        self.select_res = select_err, select_out_of_place, select_missing
        return

    # New implementation of test_having
    # def test_having_new(self, num_rs=2):
    #     # Check if both queries have HAVING
    #     if not self.q1_having_tree and not self.q2_having_tree:
    #         print("HAVING clauses are equivalent!")
    #         self.having_res = ([], [])
    #         return
    #     elif not self.q1_having_tree:
    #         print("Do not need HAVING clause")
    #         self.having_res = (
    #             [translate_query_namespace(
    #                 eval(self.q2_having_tree.getZ3()),
    #                 self.reverse_mapping[1],
    #                 self.q2_info.std_alias_to_info
    #             )],
    #             []
    #         )
    #         return
    #     elif not self.q2_having_tree:
    #         print("Missing HAVING clause")
    #         self.having_res = (
    #             [],
    #             [translate_query_namespace(
    #                 eval(self.q1_having_tree.getZ3()),
    #                 self.reverse_mapping[1],
    #                 self.q2_info.std_alias_to_info
    #             )]
    #         )
    #         return

    #     # Build context: WHERE clause (already verified correct)
    #     contexts = []
    #     if self.q1_where_tree:
    #         contexts.append(self.q1_where_tree.getZ3())

    #     # Check HAVING equivalence under WHERE constraint
    #     if (self.check_implication(self.q1_having_tree.getZ3(), self.q2_having_tree.getZ3(), contexts)
    #         and self.check_implication(self.q2_having_tree.getZ3(), self.q1_having_tree.getZ3(), contexts)):
    #         print("HAVING clauses are equivalent!")
    #         self.having_res = ([], [])
    #         return

    #     # Find repair sites and generate fixes
    #     final_fixes = None
    #     cost = None

    #     iter = RepairSitesIter(
    #         self.q2_having_tree, num_rs,
    #         self.q1_having_tree.get_size(),
    #         self.q2_having_tree.get_size()
    #     )
    #     cur_rs = iter.next()
    #     is_conj = is_conjunctive(self.q2_having_tree)

    #     while cur_rs:
    #         repair_site_sz = sum([s.get_size() for s in cur_rs]) / (
    #             self.q1_having_tree.get_size() + self.q2_having_tree.get_size()
    #         ) + 1 / 6 * get_actual_rs_count(cur_rs)

    #         if cost is not None and repair_site_sz > cost:
    #             break

    #         # Verify repair sites (bounds should be valid under WHERE constraint)
    #         # We need to check if lower -> target and target -> upper hold under WHERE
    #         lower, upper = self.create_bounds_nary(self.q2_having_tree, cur_rs)
    #         target_formula = self.q1_having_tree.getZ3()

    #         if not (self.check_implication(lower, target_formula, contexts)
    #                 and self.check_implication(target_formula, upper, contexts)):
    #             cur_rs = iter.next()
    #             continue

    #         # Generate fixes using FixGenerator
    #         fg = FixGenerator(self.q1_having_tree, self.q2_having_tree, self.z3_var, cur_rs)
    #         res = fg.get_fixes()

    #         cur_cost = repair_cost(self.q1_having_tree, self.q2_having_tree, res)

    #         if cost is None or cur_cost < cost:
    #             final_fixes = res
    #             cost = cur_cost
    #             self.fg = fg

    #         cur_rs = iter.next()

    #         if is_conj:
    #             break

    #     # Store results in having_res
    #     self.having_res = ([], [])
    #     if final_fixes:
    #         for s in final_fixes:
    #             tmp_rs = [
    #                 translate_query_namespace(
    #                     eval(x.getZ3()),
    #                     self.reverse_mapping[1],
    #                     self.q2_info.std_alias_to_info
    #                 )
    #                 for x in s[0]
    #             ]
    #             tmp_rs_sz = sum([x.get_size() for x in s[0]])
    #             tmp_f = translate_query_namespace(
    #                 s[1][0],
    #                 self.reverse_mapping[1],
    #                 self.q2_info.std_alias_to_info
    #             )
    #             tmp_f_sz = s[1][1]

    #             self.having_res[0].append(((tmp_rs, tmp_f), (tmp_rs_sz, tmp_f_sz)))

    #     # Print results
    #     self.print_rs_fixes(self.having_res[0])


    # Simplified aggregate axioms using value-level expressions (no arrays) -- REAL VERSION
    def check_implication_with_simple_agg_axioms(self, q1: str, q2: str, contexts: list[str] = []) -> bool:
        """
        Test if q1 -> q2 holds true, with simplified aggregate function axioms.
        Uses value-level axioms like SUM(a) + SUM(b) = SUM(a+b) without array modeling.

        ### Input
            q1: string, a formula ready for eval()
            q2: string, a formula ready for eval()
            contexts: list[str], additional context constraints (e.g., WHERE clause)

        ### Return
            bool, true if q1 -> q2, else false
        """

        # Create FRESH solver for each call to avoid state contamination
        solver = Solver()
        solver.set("timeout", 3000)

        # Declare symbolic variables - MUST use Real to match actual variable types
        A, B, C = Consts('agg_A agg_B agg_C', RealSort())

        # Define simplified aggregate axioms using value-level rules
        simple_agg_axioms = [
            ForAll([A, C], SUM(A * C) == SUM(A) * C),  # scalar multiplication
            ForAll([A, B], SUM(A + B) == SUM(A) + SUM(B)),  # addition
            ForAll([A, B], SUM(A - B) == SUM(A) - SUM(B)),  # subtraction
            ForAll([A, B], SUM(A * B) == SUM(A) * SUM(B)),  # multiplication
            ForAll([A, B], SUM(A / B) == SUM(A) / SUM(B)),  # division

            ForAll([A, C], AVG(A * C) == AVG(A) * C),
            ForAll([A, B], AVG(A + B) == AVG(A) + AVG(B)),
            ForAll([A, B], AVG(A - B) == AVG(A) - AVG(B)),
            ForAll([A, B], AVG(A * B) == AVG(A) * AVG(B)),
        ]

        # Add axioms to solver
        solver.add(simple_agg_axioms)

        # Add context constraints (e.g., WHERE clause)
        for ctx in contexts:
            solver.add(eval(ctx))

        # Check implication: q1 -> q2
        q1_formula = eval(q1) if type(q1) == str else q1
        q2_formula = eval(q2) if type(q2) == str else q2
        solver.add(Not(Implies(q1_formula, q2_formula)))

        result = solver.check()

        # Debug output
        print(f"[DEBUG] Checking: {q1} -> {q2}")
        print(f"[DEBUG] Result: {result}")
        if result == unknown:
            print(f"[DEBUG] WARNING: Z3 returned unknown")
            print(f"[DEBUG] Reason: {solver.reason_unknown()}")

        return result == unsat

        # # IMPORTANT: Create a fresh solver for each call to avoid state contamination
        # solver = Solver()
        # solver.set("timeout", 5000)

        # # Declare symbolic variables for aggregate operands
        # a, b = Ints('agg_a agg_b')
        # c = Int('agg_c')

        # # Define simplified aggregate axioms using value-level rules
        # simple_agg_axioms = [
        #     # SUM linearity with integers
        #     # ForAll([a, b], SUM(a + b) == SUM(a) + SUM(b)),
        #     # ForAll([a, b], SUM(a - b) == SUM(a) - SUM(b)),
        #     ForAll([a, c], SUM(a * c) == SUM(a) * c),  # scalar multiplication

        #     # AVG linearity (simplified, assumes equal group sizes)
        #     ForAll([a, b], AVG(a + b) == AVG(a) + AVG(b)),
        #     ForAll([a, b], AVG(a - b) == AVG(a) - AVG(b)),
        # ]
        # # Add axioms to solver
        # solver.add(simple_agg_axioms)

        # # Add context constraints (e.g., WHERE clause)
        # for ctx in contexts:
        #     solver.add(eval(ctx))

        # # Check implication: q1 -> q2
        # q1_formula = eval(q1) if type(q1) == str else q1
        # q2_formula = eval(q2) if type(q2) == str else q2
        # solver.add(Not(Implies(q1_formula, q2_formula)))

        # result = solver.check()

        # # Debug output
        # print(f"[DEBUG] Checking: {q1} -> {q2}")
        # print(f"[DEBUG] Result: {result}")
        # if result == unknown:
        #     print(f"[DEBUG] WARNING: Z3 returned unknown")
        #     print(f"[DEBUG] Reason: {solver.reason_unknown()}")

        # return result == unsat
        


    # Enhanced test_having that uses simplified aggregate axioms -- REAL VERSION
    def test_having_simple_agg(self, num_rs=2):
        """
        Test HAVING clause with simplified aggregate function axioms.
        Uses value-level rules like SUM(a+b) = SUM(a) + SUM(b).
        Precondition: WHERE and GROUP BY have been verified to be correct.

        Note: Non-aggregate HAVING conditions are already checked in test_where_fg,
        so here we only compare aggregate conditions (COUNT, SUM, AVG, etc.)
        """
        # Use aggregate-only HAVING trees (non-aggregate conditions already checked in WHERE stage)
        q1_having_agg = self.q1_having_agg_only
        q2_having_agg = self.q2_having_agg_only

        # Check if both queries have aggregate HAVING conditions
        if not q1_having_agg and not q2_having_agg:
            print("HAVING clauses (aggregate conditions) are equivalent!")
            self.having_res = ([], [])
            return
        elif not q1_having_agg:
            print("Do not need HAVING clause (aggregate conditions)")
            self.having_res = (
                [translate_query_namespace(
                    eval(q2_having_agg.getZ3()),
                    self.reverse_mapping[1],
                    self.q2_info.std_alias_to_info
                )],
                []
            )
            return
        elif not q2_having_agg:
            print("Missing HAVING clause (aggregate conditions)")
            self.having_res = (
                [],
                [translate_query_namespace(
                    eval(q1_having_agg.getZ3()),
                    self.reverse_mapping[1],
                    self.q2_info.std_alias_to_info
                )]
            )
            return

        # Check for DISTINCT mismatches in HAVING clause (using original trees for full context)
        if self.q1_having_tree and self.q2_having_tree:
            mismatches = self._find_aggregate_distinct_mismatches(
                self.q1_having_tree,
                self.q2_having_tree
            )
            if mismatches:
                self.distinct_mismatches["having"] = mismatches

        # Build context: WHERE clause (already verified correct)
        contexts = []
        if self.q1_where_tree:
            contexts.append(self.q1_where_tree.getZ3())

        # Check HAVING equivalence using simplified aggregate axioms
        if (self.check_implication_with_simple_agg_axioms(
                q1_having_agg.getZ3(), q2_having_agg.getZ3(), contexts)
            and self.check_implication_with_simple_agg_axioms(
                q2_having_agg.getZ3(), q1_having_agg.getZ3(), contexts)):
            print("HAVING clauses (aggregate conditions) are equivalent!")
            self.having_res = ([], [])
            return

        # Find repair sites and generate fixes
        final_fixes = None
        cost = None

        # Make aggregate-only trees n-ary for repair iteration
        q1_having_agg_nary = convert_to_nary_tree(q1_having_agg)
        q2_having_agg_nary = convert_to_nary_tree(q2_having_agg)

        iter = RepairSitesIter(
            q2_having_agg_nary, num_rs,
            q1_having_agg_nary.get_size(),
            q2_having_agg_nary.get_size()
        )
        cur_rs = iter.next()
        is_conj = is_conjunctive(q2_having_agg_nary)

        while cur_rs:
            repair_site_sz = sum([s.get_size() for s in cur_rs]) / (
                q1_having_agg_nary.get_size() + q2_having_agg_nary.get_size()
            ) + 1 / 6 * get_actual_rs_count(cur_rs)

            if cost is not None and repair_site_sz > cost:
                break

            # Verify repair sites using simplified aggregate axioms
            lower, upper = self.create_bounds_nary(q2_having_agg_nary, cur_rs)
            target_formula = q1_having_agg_nary.getZ3()

            if not (self.check_implication_with_simple_agg_axioms(lower, target_formula, contexts)
                    and self.check_implication_with_simple_agg_axioms(target_formula, upper, contexts)):
                cur_rs = iter.next()
                continue

            # Generate fixes using FixGenerator
            fg = FixGenerator(q1_having_agg_nary, q2_having_agg_nary, self.z3_var, cur_rs)
            res = fg.get_fixes()

            cur_cost = repair_cost(q1_having_agg_nary, q2_having_agg_nary, res)

            if cost is None or cur_cost < cost:
                final_fixes = res
                cost = cur_cost
                self.fg = fg

            cur_rs = iter.next()

            if is_conj:
                break

        # Store results in having_res
        self.having_res = ([], [])
        if final_fixes:
            for s in final_fixes:
                tmp_rs = [
                    translate_query_namespace(
                        eval(x.getZ3()),
                        self.reverse_mapping[1],
                        self.q2_info.std_alias_to_info
                    )
                    for x in s[0]
                ]
                tmp_rs_sz = sum([x.get_size() for x in s[0]])
                tmp_f = translate_query_namespace(
                    s[1][0],
                    self.reverse_mapping[1],
                    self.q2_info.std_alias_to_info
                )
                tmp_f_sz = s[1][1]

                self.having_res[0].append(((tmp_rs, tmp_f), (tmp_rs_sz, tmp_f_sz)))

        # Print results
        self.print_rs_fixes(self.having_res[0])



def examine_queries(q1: str, q2: str):
    """Given two query strings, invoke the entire process of hinting.
    ### Input
        q1: str,
        q2: str,
    ### Return
        (str, str)[], list of pair of repair sites and fixes
    """
    q1_info = QueryInfo(q1)
    q2_info = QueryInfo(q2)
    # Test FROM
    try:
        m = MappingInfo(q1_info, q2_info)
    except ValueError as e:
        print(e)
        return "FROM"

    t = QueryTest(q1_info, q2_info, m.z3_var_lookup, m.table_mapping, m.table_mapping_reverse)

    # now we have picked the right query mapping for test
    # Test WHERE / HAVING
    # t.test_where_having_min_overall_fg() TODO: comment out for now due to unimplemented

    
    if t.rs_fix_pair_fg:
        print(t.rs_fix_pair_fg)
        return "WHERE/HAVING"

    # Test GROUP-BY
    t.test_group_by()
    if not (t.group_by_res is None or not (t.group_by_res[0] and t.group_by_res[1])):
        return "GROUP BY"

    # Test HAVING

    # Test SELECT
    t.test_select()
    if t.select_res is None or t.select_res[0] or t.select_res[1] or t.select_res[2]:
        return "SELECT"

    return "No Error"





