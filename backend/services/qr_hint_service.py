# path: backend/services/qr_hint_service.py
# import sys
import os
import importlib

# import qr_hint function
from qr_hint.query_info import QueryInfo, MappingInfo
from qr_hint.query_test import QueryTest
from qr_hint.exists_rewriter import ExistsRewriter

# Add qr_hint directory to path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'qr_hint'))

from z3 import *


def load_schema_config(schema: str):
    """
    Dynamically load the global_var module for the specified schema.

    Args:
        schema: Schema name ('beers', 'dblp', or 'tpc')

    Returns:
        tuple: (analyzer, db_schema) for the specified schema
    """
    schema_module_map = {
        'beers': 'qr_hint.global_var_beers',
        'dblp': 'qr_hint.global_var_dblp',
        'tpc': 'qr_hint.global_var_tpc'
    }

    module_name = schema_module_map.get(schema, 'qr_hint.global_var_beers')
    module = importlib.import_module(module_name)

    return module.default_analyzer, module.db_schema


class QRHintService:
    """Service class for QR-Hint core logic"""

    def __init__(self):
        """Initialize QR-Hint service"""
        self.jar_path = os.getenv('QR_HINT_JAR_PATH', './qr_hint/sqlanalyzer.jar')
        # Import qr_hint modules when needed
        self._query_info = None
        self._fix_generator = None
        self._fix_optimizer = None

    def _lazy_load_modules(self):
        """Lazy load qr_hint modules to avoid startup overhead"""
        if self._query_info is None:
            try:
                # Import qr_hint modules
                # These will be imported when actually needed
                pass
            except ImportError as e:
                print(f"Warning: Could not import qr_hint modules: {e}")

    # def parse_query(self, query_text: str) -> dict:
    #     """
    #     Parse SQL query and return structure information

    #     Args:
    #         query_text: SQL query string

    #     Returns:
    #         Dictionary containing query structure information
    #     """
    #     # TODO: Integrate with actual qr_hint query parser
    #     # For now, return a placeholder
    #     return {
    #         'query': query_text,
    #         'parsed': True,
    #         'tables': [],
    #         'columns': [],
    #         'joins': []
    #     }

    def generate_hint(self, query_text: str, options: dict = None) -> dict:
        """
        Generate query hint based on input query

        Args:
            query_text: SQL query string
            options: Optional dictionary of hint generation options

        Returns:
            Dictionary containing generated hint information
        """
        if options is None:
            options = {}

        # TODO: Integrate with actual qr_hint hint generator
        # For now, return a placeholder
        return {
            'query': query_text,
            'hint': 'USE_INDEX(table_name index_name)',
            'confidence': 0.85,
            'suggestions': []
        }

    # def optimize_query(self, query_text: str, hint: str = None) -> dict:
    #     """
    #     Optimize query with generated hints

    #     Args:
    #         query_text: SQL query string
    #         hint: Optional hint to apply

    #     Returns:
    #         Dictionary containing optimization results
    #     """
    #     # TODO: Integrate with actual qr_hint optimizer
    #     # For now, return a placeholder
    #     optimized_query = query_text
    #     if hint:
    #         optimized_query = f"{hint}\n{query_text}"

    #     return {
    #         'original_query': query_text,
    #         'optimized_query': optimized_query,
    #         'hint_applied': hint,
    #         'improvement': '15%'
    #     }

    # def analyze_query_plan(self, query_text: str) -> dict:
    #     """
    #     Analyze query execution plan

    #     Args:
    #         query_text: SQL query string

    #     Returns:
    #         Dictionary containing query plan analysis
    #     """
    #     # TODO: Integrate with actual query plan analyzer
    #     return {
    #         'query': query_text,
    #         'plan': {},
    #         'cost': 0,
    #         'bottlenecks': []
    #     }
    
    def _simplify_error_message(self, error: Exception) -> str:
        """
        Convert technical error messages to user-friendly messages.

        Args:
            error: The exception object

        Returns:
            Simplified error message string
        """
        error_str = str(error)
        error_type = type(error).__name__

        # SQL syntax errors from QueryInfo
        if "Query syntax error" in error_str:
            # Extract the actual error message
            if ":" in error_str:
                parts = error_str.split(":")
                # return f"SQL Syntax Error: {parts[-1].strip()}"
                return f"{parts[-1].strip()}"
            return "SQL syntax error in your query. Please check the SQL syntax."

        # Not implemented features
        if isinstance(error, NotImplementedError):
            if "not supported" in error_str:
                return "Your query uses features that are not yet supported by the repair tool."
            if "inline derived table" in error_str or "WITH reference" in error_str:
                return "Subqueries and WITH clauses are not supported yet."
            return "This query feature is not supported yet."
        
        if isinstance(error, KeyError):
            return "Key Error: " + error_str

        # Table-related errors
        if isinstance(error, ValueError):
            if "missing" in error_str.lower() or "redundant" in error_str.lower():
                return f"Table Error: {error_str}"
            return error_str

        # Runtime errors from query parsing(might be processed in first if)
        if isinstance(error, RuntimeError):
            return f"Query Processing Error: {error_str.split(':', 1)[-1].strip()}"

        # Generic errors - provide helpful message
        if len(error_str) > 200:
            return "Query analysis failed. Please verify your SQL syntax."

        return error_str

    
    def _preprocess_subquery(self, query: str, analyzer=None) -> tuple:
        """
        Preprocess query containing subqueries (EXISTS, SOME/ANY).

        Args:
            query: SQL query string
            analyzer: SQL analyzer for the specific schema (optional)

        Returns:
            tuple: (rewritten_query, subquery_info)
                - rewritten_query: The query with subqueries rewritten to JOINs
                - subquery_info: dict containing transformation details
        """
        subquery_info = {
            'has_exists': False,
            'has_some': False,
            'original_query': query,
            'rewritten_query': None,
            'tables_added': []
        }

        try:
            if analyzer:
                rewriter = ExistsRewriter(query, analyzer=analyzer)
            else:
                rewriter = ExistsRewriter(query)
            has_exists = rewriter.contains_exists()
            has_some = rewriter.contains_some_any()

            if has_exists or has_some:
                rewritten = rewriter.rewrite()
                subquery_info['has_exists'] = has_exists
                subquery_info['has_some'] = has_some
                subquery_info['rewritten_query'] = rewritten
                subquery_info['tables_added'] = list(rewriter.used_table_names)
                return rewritten, subquery_info
        except Exception as e:
            # If rewrite fails, return original query with error info
            subquery_info['rewrite_error'] = str(e)

        return query, subquery_info

    # Keep old method name for backward compatibility
    def _preprocess_exists_query(self, query: str) -> tuple:
        """Backward compatible alias for _preprocess_subquery"""
        return self._preprocess_subquery(query)

    # Repair query
    def repair_query(self, correct_query: str, incorrect_query: str, schema: str = 'beers') -> dict:
        """
        Analyze two SQL queries (one correct, one incorrect) and suggest repairs

        Args:
            correct_query: The correct SQL query
            incorrect_query: The incorrect SQL query to be repaired
            schema: Database schema to use ('beers', 'dblp', or 'tpc')

        Returns:
            Dictionary containing repair suggestions
        """
        try:
            # Load the appropriate schema configuration
            analyzer, db_schema = load_schema_config(schema)

            # Preprocess both queries to detect and rewrite subqueries (EXISTS, SOME/ANY)
            # Case 1: Correct has subquery, user doesn't -> rewrite correct
            # Case 2: Correct doesn't have subquery, user does -> rewrite user
            # Case 3: Both have subqueries -> rewrite both
            # Case 4: Neither has subquery -> use as-is

            correct_query_processed, correct_subquery_info = self._preprocess_subquery(correct_query, analyzer)
            incorrect_query_processed, user_subquery_info = self._preprocess_subquery(incorrect_query, analyzer)

            # Determine which queries were rewritten
            correct_has_subquery = correct_subquery_info.get('has_exists') or correct_subquery_info.get('has_some')
            user_has_subquery = user_subquery_info.get('has_exists') or user_subquery_info.get('has_some')

            # Build combined subquery_info for frontend display
            subquery_info = {
                'has_exists': user_subquery_info.get('has_exists', False) or correct_subquery_info.get('has_exists', False),
                'has_some': user_subquery_info.get('has_some', False) or correct_subquery_info.get('has_some', False),
                'original_query': incorrect_query,
                'rewritten_query': user_subquery_info.get('rewritten_query'),
                'tables_added': user_subquery_info.get('tables_added', []),
                # New fields to track both sides
                'user_has_subquery': user_has_subquery,
                'correct_has_subquery': correct_has_subquery,
                'correct_original': correct_query if correct_has_subquery else None,
                'correct_rewritten': correct_subquery_info.get('rewritten_query') if correct_has_subquery else None,
                'subquery_style_mismatch': correct_has_subquery != user_has_subquery
            }

            # Create QueryInfo objects for both queries (using processed versions)
            q1_info = QueryInfo(correct_query_processed, analyzer=analyzer, schema=db_schema)
            q2_info = QueryInfo(incorrect_query_processed, analyzer=analyzer, schema=db_schema)

            # Check FROM clause first by trying to create MappingInfo
            # mapping_check() is called in MappingInfo.__init__
            from_missing = []
            from_redundant = []
            from_wrong_count = []
            from_tested = True
            has_from_issues = False
            m = None

            try:
                # Create mapping between queries
                # This will call mapping_check() internally
                m = MappingInfo(q1_info, q2_info, schema=db_schema)
            except ValueError as e:
                # Parse the error message to extract FROM issues
                error_msg = str(e)
                has_from_issues = True

                if "are missing and" in error_msg and "are redundant" in error_msg:
                    # Both missing and redundant tables
                    parts = error_msg.split("are missing and Table")
                    if len(parts) == 2:
                        missing_part = parts[0].replace("Table", "").strip()
                        redundant_part = parts[1].replace("are redundant.", "").strip()
                        # Extract table names from sets like "{'Table1', 'Table2'}"
                        from_missing = [t.strip().strip("'\"") for t in missing_part.strip("{}").split(",") if t.strip()]
                        from_redundant = [t.strip().strip("'\"") for t in redundant_part.strip("{}").split(",") if t.strip()]
                elif "are missing" in error_msg:
                    # Only missing tables
                    missing_part = error_msg.replace("Table", "").replace("are missing.", "").strip()
                    from_missing = [t.strip().strip("'\"") for t in missing_part.strip("{}").split(",") if t.strip()]
                elif "are redundant" in error_msg:
                    # Only redundant tables
                    redundant_part = error_msg.replace("Table", "").replace("are redundant.", "").strip()
                    from_redundant = [t.strip().strip("'\"") for t in redundant_part.strip("{}").split(",") if t.strip()]
                elif "should be referenced" in error_msg:
                    # Wrong count of table references
                    from_wrong_count.append(error_msg)
                else:
                    # Unknown FROM error format
                    from_wrong_count.append(error_msg)
            # Only proceed to WHERE and GROUP BY if FROM is correct
            repair_results = []
            group_by_incorrect = []
            group_by_missing = []
            group_by_tested = False
            has_where_issues = False
            has_group_by_issues = False

            # Initialize HAVING variables
            having_repairs = []
            having_distinct_mismatches = []
            having_tested = False
            has_having_issues = False

            # Initialize SELECT variables
            select_incorrect = []
            select_wrong_order = []
            select_missing_items = []
            select_distinct_mismatches = []
            select_tested = False
            has_select_issues = False

            if not has_from_issues and m is not None:
                # FROM is correct, create QueryTest object
                t = QueryTest(q1_info, q2_info, m.z3_var_lookup, m.table_mapping, m.table_mapping_reverse, schema=db_schema)

                # Run the WHERE clause test (test_where_fg)
                t.test_where_fg()

                # Extract multiple repair solutions from the results
                # Each solution contains cost and list of repairs
                if t.rs_fix_pair_fg:
                    for solution_idx, solution_data in enumerate(t.rs_fix_pair_fg):
                        solution_cost = solution_data['cost']
                        solution_repairs = []

                        for repair_idx, (rs_f, sizes) in enumerate(solution_data['repairs']):
                            repair_site = rs_f[0]
                            fix = rs_f[1]
                            repair_site_size = sizes[0]
                            fix_size = sizes[1]

                            # Convert fix to user-friendly string
                            # When fix is True (Python bool) or "True" (string), it means
                            # the condition should be removed (replaced with True = no constraint)
                            if fix is True or str(fix) == "True":
                                fix_str = "N/A (remove this condition)"
                            elif fix is False or str(fix) == "False":
                                fix_str = "N/A (condition always false)"
                            elif fix is None:
                                fix_str = None
                            else:
                                fix_str = str(fix)

                            solution_repairs.append({
                                'repair_site_index': repair_idx,
                                'repair_site': str(repair_site) if repair_site else None,
                                'fix': fix_str,
                                'repair_site_size': repair_site_size,
                                'fix_size': fix_size
                            })

                        repair_results.append({
                            'solution_index': solution_idx,
                            'cost': solution_cost,
                            'repairs': solution_repairs
                        })

                has_where_issues = len(repair_results) > 0

                # Only run GROUP BY test if WHERE clause is correct
                if not has_where_issues:
                    # WHERE is correct, proceed to GROUP BY test
                    t.test_group_by()
                    group_by_tested = True

                    if hasattr(t, 'group_by_res') and t.group_by_res:
                        incorrect, missing = t.group_by_res
                        # Convert to strings
                        group_by_incorrect = [str(item) for item in incorrect] if incorrect else []
                        group_by_missing = [str(item) for item in missing] if missing else []
                        has_group_by_issues = len(group_by_incorrect) > 0 or len(group_by_missing) > 0

                    # Only run HAVING test if GROUP BY is correct
                    if not has_group_by_issues:
                        # GROUP BY is correct, proceed to HAVING test
                        t.test_having_simple_agg()
                        having_tested = True

                        if hasattr(t, 'having_res') and t.having_res:
                            # having_res format depends on the case:
                            # - Normal repairs: ([(repair_data, sizes), ...], [])
                            # - Missing HAVING: ([], [missing_condition])
                            # - Redundant HAVING: ([redundant_condition], [])
                            repairs_or_redundant = t.having_res[0]
                            missing_conditions = t.having_res[1]

                            # Helper function to convert fix to user-friendly string
                            def format_fix(fix):
                                if fix is True or str(fix) == "True":
                                    return "N/A (remove this condition)"
                                elif fix is False or str(fix) == "False":
                                    return "N/A (condition always false)"
                                elif fix is None:
                                    return None
                                else:
                                    return str(fix)

                            # Handle redundant/incorrect HAVING conditions (first element)
                            for repair_item in repairs_or_redundant:
                                # Check if this is a repair tuple or just a condition
                                if isinstance(repair_item, tuple) and len(repair_item) == 2:
                                    first_elem, second_elem = repair_item
                                    # Check if it's ((repair_sites, fix), (sizes)) format
                                    if isinstance(first_elem, tuple) and isinstance(second_elem, tuple):
                                        repair_sites, fix = first_elem
                                        repair_site_size, fix_size = second_elem
                                        repair_site_str = ' AND '.join([str(rs) for rs in repair_sites]) if repair_sites else None
                                        having_repairs.append({
                                            'repair_site': repair_site_str,
                                            'fix': format_fix(fix),
                                            'repair_site_size': repair_site_size,
                                            'fix_size': fix_size
                                        })
                                    else:
                                        # It's a simple condition that should be removed (redundant)
                                        having_repairs.append({
                                            'repair_site': str(repair_item),
                                            'fix': "N/A (remove this condition)",
                                            'repair_site_size': 1,
                                            'fix_size': 0
                                        })
                                else:
                                    # It's a simple condition that should be removed (redundant)
                                    having_repairs.append({
                                        'repair_site': str(repair_item),
                                        'fix': "N/A (remove this condition)",
                                        'repair_site_size': 1,
                                        'fix_size': 0
                                    })

                            # Handle missing HAVING conditions (second element)
                            for missing_item in missing_conditions:
                                having_repairs.append({
                                    'repair_site': None,  # Nothing to replace
                                    'fix': str(missing_item),  # This should be added
                                    'repair_site_size': 0,
                                    'fix_size': 1
                                })

                            has_having_issues = len(having_repairs) > 0

                        # Check for HAVING DISTINCT mismatches
                        if hasattr(t, 'distinct_mismatches') and t.distinct_mismatches:
                            having_distinct_mismatches = t.distinct_mismatches.get('having', [])
                            # Skip DISTINCT check if correct query originally used EXISTS
                            if correct_has_subquery:
                                having_distinct_mismatches = []
                            elif len(having_distinct_mismatches) > 0:
                                has_having_issues = True

                        # Only run SELECT test if HAVING is correct
                        if not has_having_issues:
                            # HAVING is correct, proceed to SELECT test
                            t.test_select()
                            select_tested = True

                            if hasattr(t, 'select_res') and t.select_res:
                                # select_res format: (select_err, select_out_of_place, select_missing)
                                select_err, select_out_of_place, select_missing = t.select_res

                                # Convert to strings
                                select_incorrect = [str(item) for item in select_err] if select_err else []
                                select_wrong_order = [str(item) for item in select_out_of_place] if select_out_of_place else []
                                select_missing_items = [str(item) for item in select_missing] if select_missing else []

                                has_select_issues = (len(select_incorrect) > 0 or
                                                    len(select_wrong_order) > 0 or
                                                    len(select_missing_items) > 0)

                            # Check for DISTINCT mismatches
                            if hasattr(t, 'distinct_mismatches') and t.distinct_mismatches:
                                select_distinct_mismatches = t.distinct_mismatches.get('select', [])
                                # Skip DISTINCT check if correct query originally used EXISTS —
                                # the DISTINCT came from the rewrite, not the original intent.
                                # Clear the list so the frontend does not display it either.
                                if correct_has_subquery:
                                    select_distinct_mismatches = []
                                elif len(select_distinct_mismatches) > 0:
                                    has_select_issues = True
                            else:
                                select_distinct_mismatches = []

            # Determine current stage and completed stages
            completed_stages = []
            if not has_from_issues:
                completed_stages.append('from')
                if not has_where_issues:
                    completed_stages.append('where')
                    if not has_group_by_issues and group_by_tested:
                        completed_stages.append('group_by')
                        if not has_having_issues and having_tested:
                            completed_stages.append('having')
                            if not has_select_issues and select_tested:
                                completed_stages.append('select')

            # Determine current stage
            if has_from_issues:
                current_stage = 'from'
            elif has_where_issues:
                current_stage = 'where'
            elif has_group_by_issues:
                current_stage = 'group_by'
            elif has_having_issues:
                current_stage = 'having'
            elif has_select_issues:
                current_stage = 'select'
            else:
                current_stage = 'complete'

            return {
                'ok': True,
                'correct_query': correct_query,
                'incorrect_query': incorrect_query,
                'subquery_info': subquery_info,  # Subquery (EXISTS/SOME/ANY) rewrite information
                'repairs': repair_results, # where repairs(refactor later)
                'repair_count': len(repair_results), # where repair count(refactor later)
                'from_clause': {
                    'missing': from_missing,
                    'redundant': from_redundant,
                    'wrong_count': from_wrong_count,
                    'has_issues': has_from_issues,
                    'tested': from_tested
                },
                'group_by': {
                    'incorrect': group_by_incorrect,
                    'missing': group_by_missing,
                    'has_issues': has_group_by_issues,
                    'tested': group_by_tested
                },
                'having': {
                    'repairs': having_repairs,
                    'distinct_mismatches': having_distinct_mismatches,
                    'has_issues': has_having_issues,
                    'tested': having_tested
                },
                'select': {
                    'incorrect': select_incorrect,
                    'wrong_order': select_wrong_order,
                    'missing': select_missing_items,
                    'distinct_mismatches': select_distinct_mismatches,
                    'has_issues': has_select_issues,
                    'tested': select_tested
                },
                'stage': {
                    'current': current_stage,
                    'completed': completed_stages,
                    'total_stages': 5
                }
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            # Simplify error message for frontend display
            simplified_error = self._simplify_error_message(e)

            return {
                'ok': False,
                'error': simplified_error,
                'error_type': type(e).__name__,
                'correct_query': correct_query,
                'incorrect_query': incorrect_query
            }
