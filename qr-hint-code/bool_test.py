# built-in packages
from copy import deepcopy

# extra packages
from z3 import *
from query_info import *

# project packages
from boolean_parse_tree import BNode
from utils import *
# from global_var_beers import *
# from global_var_tpc import *
# from global_var_dblp import *
from subtree_iter import *
from fix_generator import FixGenerator
from fix_optimizer import FixOptimizer
import time


COUNT = Function('COUNT', IntSort(), IntSort())
MAX = Function('MAX', IntSort(), IntSort())
MIN = Function('MIN', IntSort(), IntSort())
AVG = Function('AVG', IntSort(), IntSort())
SUM = Function('SUM', IntSort(), IntSort())


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
    def __init__(self, q1_info: QueryInfo, q2_info: QueryInfo, z3_lookup: dict, mapping, reverse_mapping, schema=db_schema):
        self.schema = schema
        self.q1_info = q1_info
        self.q2_info = q2_info
        
        self.z3_var = z3_lookup
        self.z3_var_g = {}
        self.mapping = mapping
        self.reverse_mapping = reverse_mapping
        self.solver = Solver()

        self.q1_where_tree = self.build_syntax_tree(self.q1_info.flatten_where_trees, self.mapping[0], self.q1_info.attr_trace, 'self.z3_var')
        self.q2_where_tree = self.build_syntax_tree(self.q2_info.flatten_where_trees, self.mapping[1], self.q2_info.attr_trace, 'self.z3_var')

        self.q1_groupby_expr = [self.build_syntax_tree(x, self.mapping[0], self.q1_info.attr_trace, 'self.z3_var') for x in self.q1_info.flatten_groupby_exprs] if q1_info.flatten_groupby_exprs else []
        self.q2_groupby_expr = [self.build_syntax_tree(x, self.mapping[1], self.q2_info.attr_trace, 'self.z3_var') for x in self.q2_info.flatten_groupby_exprs] if q2_info.flatten_groupby_exprs else []

        self.q1_having_tree = self.build_syntax_tree(self.q1_info.flatten_having, self.mapping[0], self.q1_info.attr_trace, 'self.z3_var')
        self.q2_having_tree = self.build_syntax_tree(self.q2_info.flatten_having, self.mapping[1], self.q2_info.attr_trace, 'self.z3_var')

        self.q1_select_expr = [self.build_syntax_tree(x, self.mapping[0], self.q1_info.attr_trace, 'self.z3_var') for x in self.q1_info.flatten_select] if self.q1_info.flatten_select else []
        self.q2_select_expr = [self.build_syntax_tree(x, self.mapping[1], self.q2_info.attr_trace, 'self.z3_var') for x in self.q2_info.flatten_select] if self.q2_info.flatten_select else []

        # Combine where and having tree, this is the old approach
        # self.q1_comb_tree, self.q2_comb_tree = None, None
        # if self.q1_where_tree and self.q1_having_tree:
        #     self.q1_comb_tree = BNode('And', 'log', select_xid=self.q1_where_tree.select_xid, parent=None)
        #     self.q1_comb_tree.children = [self.q1_where_tree, self.q1_having_tree]
        # elif self.q1_where_tree:
        #     self.q1_comb_tree = self.q1_where_tree
        # elif self.q1_having_tree:
        #     self.q1_comb_tree = self.q1_having_tree

        # if self.q2_where_tree and self.q2_having_tree:
        #     self.q2_comb_tree = BNode('And', 'log', select_xid=self.q2_where_tree.select_xid, parent=None)
        #     self.q2_comb_tree.children = [self.q2_where_tree, self.q2_having_tree]
        # elif self.q2_where_tree:
        #     self.q2_comb_tree = self.q2_where_tree
        # elif self.q2_having_tree:
        #     self.q2_comb_tree = self.q2_having_tree

        # Combine where and having tree, new approach that uses cnf
        self.q1_comb_tree, self.q2_comb_tree = None, None
        if self.q1_where_tree and self.q1_having_tree:
            self.q1_comb_tree = BNode('And', 'log', select_xid=self.q1_where_tree.select_xid, parent=None)
            self.q1_comb_tree.children = [self.q1_where_tree, get_non_agg_cnf_clauses(convert_binary_to_cnf(self.q1_having_tree))]
        elif self.q1_where_tree:
            self.q1_comb_tree = self.q1_where_tree
        elif self.q1_having_tree:
            self.q1_comb_tree = self.q1_having_tree

        if self.q2_where_tree and self.q2_having_tree:
            self.q2_comb_tree = BNode('And', 'log', select_xid=self.q2_where_tree.select_xid, parent=None)
            self.q2_comb_tree.children = [self.q2_where_tree, get_non_agg_cnf_clauses(convert_binary_to_cnf(self.q2_having_tree))]
        elif self.q2_where_tree:
            self.q2_comb_tree = self.q2_where_tree
        elif self.q2_having_tree:
            self.q2_comb_tree = self.q2_having_tree


        # make sure both trees are n-ary
        self.q1_comb_tree = convert_to_nary_tree(self.q1_comb_tree)
        self.q2_comb_tree = convert_to_nary_tree(self.q2_comb_tree)

        self.mapping_dist = calculate_normalized_distance(eval(self.q1_comb_tree.getZ3()), eval(self.q2_comb_tree.getZ3()))

        # TODO: construct z3 formulas for HAVING clauses

        # Test results are saved here
        self.tt = None
        self.fg = None
        self.fo = None
        self.rs_fix_pair_ttg = []
        self.rs_fix_pair_fg = []
        self.rs_fix_pair_fo = []


        # Test data collection: time[], cost[], first rs time
        self.fg_converge = [[], []]
        self.fo_converge = [[], []]


    # =============================================================
    # Setup/util functions section
    # =============================================================
    def build_syntax_tree(self, xnode, alias_to_mutual, attr_trace, z3_dict, parent=None):
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

        select_xid = xnode['xid'] if 'xid' in xnode else None

        if xnode['type'] == 'XBasicCallNode' or xnode['type'] == 'XConnector':
            if xnode['operator_name'] in log_ops or xnode['operator_name'] in cmp_ops:
                op_type = 'log' if xnode['operator_name'].upper() in log_ops else 'pred'
                n = BNode(xnode['operator_name'].capitalize(), op_type, select_xid, parent)
                if xnode['operator_name'].upper() == 'NOT':
                    n.children.append(self.build_syntax_tree(xnode['operands'][0], alias_to_mutual, attr_trace, z3_dict, n))
                else:
                    n.children.append(self.build_syntax_tree(xnode['operands'][0], alias_to_mutual, attr_trace, z3_dict, n))
                    n.children.append(self.build_syntax_tree(xnode['operands'][1], alias_to_mutual, attr_trace, z3_dict, n))
                return n

            elif xnode['operator_name'] in ari_ops:
                op_type = 'expr'
                op_val = '+' if xnode['operator_name'] == '||' else xnode['operator_name']
                n = BNode(op_val, op_type, select_xid, parent)
                n.children.append(self.build_syntax_tree(xnode['operands'][0], alias_to_mutual, attr_trace, z3_dict, n))
                n.children.append(self.build_syntax_tree(xnode['operands'][1], alias_to_mutual, attr_trace, z3_dict, n))
                return n

            elif xnode['operator_name'] in agg_ops:
                op_type, op_val = 'expr', xnode['operator_name']
                n = BNode(op_val, op_type, select_xid, parent)
                n.children.append(self.build_syntax_tree(xnode['operands'][0], alias_to_mutual, attr_trace, z3_dict, n))
                return n

            else:
                # don't handle this operator
                raise NotImplementedError(f'Error building syntax tree: Operator < {xnode["operator_name"]} > is not currently supported.')

        elif xnode['type'] == 'XColumnRefNode':
            # Double check to make sure form valid z3 formula
            attr_split = xnode['sql_string'].split('.')
            table_alias, column_alias = self.trace_std_alias(xnode['XSelectNode_id'], attr_split[0], attr_split[1], attr_trace)
            table_alias = alias_to_mutual[table_alias]
            table = table_alias.split('_')[0]
            col_idx = self.schema[table][1].index(column_alias)
            return BNode(f'{z3_dict}["{table_alias}"][{col_idx}]', 'var', select_xid, parent)

        elif xnode['type'] == 'XLiteralNode':
            return BNode(xnode['sql_string'], 'const', select_xid, parent)

        elif xnode['type'] == 'XColumnRenameNode':
            return self.build_syntax_tree(xnode['operand'], alias_to_mutual, attr_trace, z3_dict, parent)
        
        # Node type not supported
        raise NotImplementedError(f'Building Syntax Tree Error: Do not support node type {xnode["type"]}.')

    
    def trace_std_alias(self, select_xid, table_alias, col, attr_trace):
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

        while next_attr[-1] != 'std':
            next_attr = attr_trace[(next_attr[0], next_attr[1], next_attr[2])]
            if next_attr[-1] == 'expr':
                raise ValueError(f'bool_test: trace_std_alias: column ({next_attr[0], next_attr[1]}) did not expand properly.')

        return next_attr[0], next_attr[1]
    

    def check_implication(self, q1, q2):
        """Test if q1 -> q2 holds true
        ### Input
            q1: string, a formula ready for eval()
            q2: string, a formula ready for eval()

        ### Return
            bool, true of q1 -> q2, else false
        """
        q1 = eval(q1) if type(q1) == str else q1
        q2 = eval(q2) if type(q2) == str else q2

        self.solver.reset()
        self.solver.add(Not(Implies(q1, q2)))
        res = self.solver.check()
        
        return res == unsat
    

    def create_bounds_nary(self, syn_tree: BNode, disjoint_trees: list):
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
            syn_tree.bounds = ('False', 'True')
            return syn_tree.bounds
        elif not syn_tree.children or syn_tree.type == 'pred':
            syn_tree.bounds = (syn_tree.getZ3(), syn_tree.getZ3())
            return syn_tree.bounds

        final_bounds = None

        if len(syn_tree.children) == 1:
            child_bounds = self.create_bounds_nary(syn_tree.children[0], disjoint_trees)
            final_bounds = [f'{syn_tree.val}({child_bounds[1]})', f'{syn_tree.val}({child_bounds[0]})']
        elif len(syn_tree.children) > 1:
            child0_bounds = self.create_bounds_nary(syn_tree.children[0], disjoint_trees)
            child1_bounds = self.create_bounds_nary(syn_tree.children[1], disjoint_trees)
            final_bounds = [f'{syn_tree.val}({child0_bounds[0]}, {child1_bounds[0]})',
                            f'{syn_tree.val}({child0_bounds[1]}, {child1_bounds[1]})']
            for i in range(2, len(syn_tree.children)):
                tmp = self.create_bounds_nary(syn_tree.children[i], disjoint_trees)
                final_bounds[0] = f'{syn_tree.val}({final_bounds[0]}, {tmp[0]})'
                final_bounds[1] = f'{syn_tree.val}({final_bounds[1]}, {tmp[1]})'

        syn_tree.bounds = tuple(final_bounds)
        return syn_tree.bounds


    def verify_repair_sites(self, disjoint_trees):
        """Verify if a set of disjoint trees is a set of repair sites.
        ### Input
            disjoint_trees: BNode[], a set of distjoint trees
        ### Return
            bool, true if valid repair sites, else false
        """
        lower, upper = self.create_bounds_nary(self.q2_comb_tree, disjoint_trees)
        target_formula = self.q1_comb_tree.getZ3()

        if self.check_implication(lower, target_formula) and self.check_implication(target_formula, upper):
            return True

        return False


    def find_smallest_rs(self, root: BNode):
        """Find the smallest (in terms of total size) set of repair sites.
        ### Input
            root: BNode, root of the syntax tree
        ### Return
            BNode[], a set of repair sites
        """
        iter = RepairSitesIter(self.q2_comb_tree, root, self.q1_comb_tree.get_size(), self.q2_comb_tree.get_size())
        subtrees = iter.next()

        while subtrees:
            if self.verify_repair_sites(subtrees):
                return subtrees     # got results

            subtrees = iter.next()

        return []
    

    # =============================================================
    # Test fucntions section.
    # =============================================================
    def test_where_having_min_overall_fo(self, num_rs=2):
        # Check where equivalence
        if self.check_implication(self.q1_comb_tree.getZ3(), self.q2_comb_tree.getZ3()) and \
            self.check_implication(self.q2_comb_tree.getZ3(), self.q1_comb_tree.getZ3()):
            print('WHERE/HAVING clauses are equivalent!')
            return

        final_fixes = None
        cost = None

        start = time.time()

        iter = RepairSitesIter(self.q2_comb_tree, num_rs, self.q1_comb_tree.get_size(), self.q2_comb_tree.get_size())
        cur_rs = iter.next()
        is_conj = is_conjunctive(self.q2_comb_tree)

        while cur_rs:
            # print('cur_rs len: ', len(cur_rs))
            repair_site_sz = sum([s.get_size() for s in cur_rs]) / (self.q1_comb_tree.get_size() + self.q2_comb_tree.get_size()) + 1 / 6 * get_actual_rs_count(cur_rs)
            if cost is not None and repair_site_sz > cost:
                break

            if not self.verify_repair_sites(cur_rs):
                cur_rs = iter.next()
                continue

            if cost == None:
                self.fo_converge.append(round(time.time() - start, 2))

            target_rs, lower, upper = lca_nary(self.q2_comb_tree, self.q1_comb_tree.getZ3(), self.q1_comb_tree.getZ3())
            # print(eval(target_rs.getZ3()))
            fo = FixOptimizer(eval(lower), eval(upper), target_rs, cur_rs, self.z3_var)
            cur_cost = repair_cost(self.q1_comb_tree, self.q2_comb_tree, fo.get_fixes())
            if cost == None or cur_cost < cost:
                final_fixes = fo.get_fixes()
                cost = cur_cost
                self.fo = fo
            cur_rs = iter.next()

            # record
            self.fo_converge[0].append(round(time.time() - start, 2))
            self.fo_converge[1].append(round(cur_cost, 2))

            if is_conj:
                break

        self.fo_min_cost = cost
        for s in final_fixes:
            tmp_rs = [translate_query_namespace(eval(x.getZ3()), self.reverse_mapping[1], self.q2_info.std_alias_to_info) for x in s[0]]
            tmp_rs_sz = sum([x.get_size() for x in s[0]])
            tmp_f = translate_query_namespace(s[1][0], self.reverse_mapping[1], self.q2_info.std_alias_to_info)
            tmp_f_sz = s[1][1]

            self.rs_fix_pair_fo.append(((tmp_rs, tmp_f), (tmp_rs_sz, tmp_f_sz)))
        self.print_rs_fixes(self.rs_fix_pair_fo)


    def test_where_having_min_overall_fg(self, num_rs=2):
        # Check where equivalence
        if self.check_implication(self.q1_comb_tree.getZ3(), self.q2_comb_tree.getZ3()) and \
            self.check_implication(self.q2_comb_tree.getZ3(), self.q1_comb_tree.getZ3()):
            print('WHERE/HAVING clauses are equivalent!')
            return

        final_fixes = None
        cost = None

        start = time.time()

        iter = RepairSitesIter(self.q2_comb_tree, num_rs, self.q1_comb_tree.get_size(), self.q2_comb_tree.get_size())
        cur_rs = iter.next()
        is_conj = is_conjunctive(self.q2_comb_tree)

        while cur_rs:
            repair_site_sz = sum([s.get_size() for s in cur_rs]) / (self.q1_comb_tree.get_size() + self.q2_comb_tree.get_size()) + 1 / 6 * get_actual_rs_count(cur_rs)
            if cost is not None and repair_site_sz > cost: # or repair_site_sz > self.q2_comb_tree.get_size() / 2:
                break

            if not self.verify_repair_sites(cur_rs):
                cur_rs = iter.next()
                continue

            if cost == None:
                self.fg_converge.append(round(time.time() - start, 2))

            fg = FixGenerator(self.q1_comb_tree, self.q2_comb_tree, self.z3_var, cur_rs)
            res = fg.get_fixes()
            
            cur_cost = repair_cost(self.q1_comb_tree, self.q2_comb_tree, res)

            if cost == None or cur_cost < cost:
                final_fixes = res
                cost = cur_cost
                self.fg = fg
            cur_rs = iter.next()
        
            # record
            self.fg_converge[0].append(round(time.time() - start, 2))
            self.fg_converge[1].append(round(cur_cost, 2))

            if is_conj:
                print('query is conjunctive')
                break

        self.fg_min_cost = cost
        for s in final_fixes:
            tmp_rs = [translate_query_namespace(eval(x.getZ3()), self.reverse_mapping[1], self.q2_info.std_alias_to_info) for x in s[0]]
            tmp_rs_sz = sum([x.get_size() for x in s[0]])
            tmp_f = translate_query_namespace(s[1][0], self.reverse_mapping[1], self.q2_info.std_alias_to_info)
            tmp_f_sz = s[1][1]

            self.rs_fix_pair_fg.append(((tmp_rs, tmp_f), (tmp_rs_sz, tmp_f_sz)))

        self.print_rs_fixes(self.rs_fix_pair_fg)


    def print_rs_fixes(self, rs_fix_pair):#: list(tuple(tuple(list(BoolRef), BoolRef), tuple(int, int)))):
        for i, (rs_f, sizes) in enumerate(rs_fix_pair):
            # for i, (rs, f) in enumerate(rs_f):
            print(f'Repair Site #{i}: {rs_f[0]}')
            # print(f'Total repair site size #{i}: {sizes[0]}')
            print(f'Fix #{i}: {rs_f[1]}')
            # print(f'Fix Size #{i}: {sizes[1]}')


    def test_group_by(self):
        # prepare for group by check
        if not self.q1_groupby_expr and self.q2_groupby_expr:
            print('Should not have any group by expressions.')
            return [], []
        elif self.q1_groupby_expr and not self.q2_groupby_expr:
            print('Need to use group by in your query.')
            return [], []
        
        # buid z3_var_g to check group by
        for key, value in self.z3_var.items():
            self.z3_var_g[key] = []
            for v in value:
                ty = str(v.sort())
                if ty == 'Int':
                    self.z3_var_g[key].append(Int(f'{str(v)}.g'))
                elif ty == 'String':
                    self.z3_var_g[key].append(String(f'{str(v)}.g'))
                elif ty == 'Real':
                    self.z3_var_g[key].append(Real(f'{str(v)}.g'))

        # clone the boolean formulae
        q1where_g = self.build_syntax_tree(self.q1_info.flatten_where_trees, self.mapping[0], self.q1_info.attr_trace, 'self.z3_var_g')
        # q2where_g = self.build_syntax_tree(self.q2_info.flatten_where_trees, self.mapping[1], self.q2_info.attr_trace, 'self.z3_var_g')
        q1having_g = self.build_syntax_tree(self.q1_info.flatten_having, self.mapping[0], self.q1_info.attr_trace, 'self.z3_var_g')
        # q2having_g = self.build_syntax_tree(self.q2_info.flatten_having, self.mapping[1], self.q2_info.attr_trace, 'self.z3_var_g')
        if q1where_g and q1having_g:
            q1comb_g = BNode('And', 'log', select_xid=self.q1_where_tree.select_xid, parent=None)
            q1comb_g.children = [q1where_g, q1having_g]
        elif q1where_g:
            q1comb_g = q1where_g
        elif q1having_g:
            q1comb_g = q1having_g

        # if q2where_g and q2having_g:
        #     q2comb_g = BNode('And', 'log', select_xid=self.q2_where_tree.select_xid, parent=None)
        #     q2comb_g.children = [q2where_g, q2having_g]
        # elif q2where_g:
        #     q2comb_g = q2where_g
        # elif q2having_g:
        #     q2comb_g = q2having_g

        # combine two boolean formulae
        q1exp = BNode('And', 'log')
        q2exp = BNode('And', 'log')
        q1exp.children = [self.q1_comb_tree, q1comb_g]
        q2exp.children = [self.q1_comb_tree, q1comb_g]

        # clone the group by expression
        q1g = [self.build_syntax_tree(x, self.mapping[0], self.q1_info.attr_trace, 'self.z3_var_g') for x in self.q1_info.flatten_groupby_exprs]
        q2g = [self.build_syntax_tree(x, self.mapping[1], self.q2_info.attr_trace, 'self.z3_var_g') for x in self.q2_info.flatten_groupby_exprs]
    
        # construct group by predicates
        q1gexp, q2gexp = [], []
        for i in range(len(q1g)):
            n = BNode('=', 'pred')
            n.children = [self.q1_groupby_expr[i], q1g[i]]
            q1gexp.append(n)
        for i in range(len(q2g)):
            n = BNode('=', 'pred')
            n.children = [self.q2_groupby_expr[i], q2g[i]]
            q2gexp.append(n)

        # parse group by into trees
        q1gt, q2gt = None, None
        if len(q1gexp) == 1:
            q1gt = q1gexp[0]
        else:
            q1gt = BNode('And', 'log')
            q1gt.children = [q1gexp[0], q1gexp[1]]
            for i in range(2, len(q1gexp)):
                tp = q1gt
                q1gt = BNode('And', 'log')
                q1gt.children = [tp, q1gexp[i]]
        
        if len(q2gexp) == 1:
            q2gt = q2gexp[0]
        else:
            q2gt = BNode('And', 'log')
            q2gt.children = [q2gexp[0], q2gexp[1]]
            for i in range(2, len(q2gexp)):
                tp = q2gt
                q2gt = BNode('And', 'log')
                q2gt.children = [tp, q2gexp[i]]

        # conjunct with boolean formula
        q1final = BNode('And', 'log')
        q1final.children = [q1gt, q1exp]
        q1final_z3 = q1final.getZ3()
        q2final = BNode('And', 'log')
        q2final.children = [q2gt, q2exp]
        q2final_z3 = q2final.getZ3()

        # make sure they are not equivalent
        if self.check_implication(q1final_z3, q2final_z3) and self.check_implication(q2final_z3, q1final_z3):
            print('GROUP BY clauses are equivalent.')
            return [], []

        missing, incorrect = [], []
        for i in range(len(q2gexp)):
            tp = BNode('And', 'log')
            tp.children = [q2exp, q2gexp[i]]
            if not self.check_implication(q1final_z3, tp.getZ3()):
                incorrect.append(translate_query_namespace(eval(self.q2_groupby_expr[i].getZ3()), self.reverse_mapping[1], self.q2_info.std_alias_to_info))

        for i in range(len(q1gexp)):
            tp = BNode('And', 'log')
            tp.children = [q1exp, q1gexp[i]]
            if not self.check_implication(q2final_z3, tp.getZ3()):
                missing.append(translate_query_namespace(eval(self.q1_groupby_expr[i].getZ3()), self.reverse_mapping[1], self.q2_info.std_alias_to_info))

        return incorrect, missing
        
    
    def test_permute_rs_time(self):
        start = time.time()
        iter = RepairSitesIter(self.q2_comb_tree, 1, self.q1_comb_tree.get_size(), self.q2_comb_tree.get_size())
        subtrees = iter.next()
        cnt = 0
        while subtrees:
            if self.verify_repair_sites(subtrees):
                cnt += 1
            subtrees = iter.next()

        iter = RepairSitesIter(self.q2_comb_tree, 2, self.q1_comb_tree.get_size(), self.q2_comb_tree.get_size())
        subtrees = iter.next()
        while subtrees:
            if self.verify_repair_sites(subtrees):
                cnt += 1
            subtrees = iter.next()
        end = time.time()
        return cnt, end - start


    def test_select(self):
        select_err = []
        select_missing_idx = [True for i in range(len(self.q1_select_expr))]
        select_out_of_place_idx = [True for i in range(len(self.q2_select_expr))]

        for i in range(len(self.q2_select_expr)):
            satisfied = False
            for j in range(len(self.q1_select_expr)):
                # TODO: probably won't be able to support cmp operators in expr, need to check this
                rhs = f'{self.q1_select_expr[j].getZ3()} == {self.q2_select_expr[i].getZ3()}'
                
                # might have type mismatch, simply continue if that's the case
                try:
                    if self.check_implication(eval(self.q1_where_tree.getZ3()), eval(rhs)):
                        select_missing_idx[j] = False
                        satisfied = True
                        if i == j:
                            select_out_of_place_idx[i] = False
                        break
                except Exception as e:
                    continue
                    
            if not satisfied:
                select_out_of_place_idx[i] = False
                select_err.append(translate_query_namespace(eval(self.q2_select_expr[i].getZ3())),
                                                            self.reverse_mapping[1], self.q2_info.std_alias_to_info)

        select_missing, select_out_of_place = [], []
        for i in range(len(self.q1_select_expr)):
            if select_missing_idx[i]:
                select_missing.append(translate_query_namespace(eval(self.q1_select_expr[i].getZ3())), 
                                                                self.reverse_mapping[0], self.q1_info.std_alias_to_info)
        
        for i in range(len(self.q2_select_expr)):
            if select_out_of_place_idx[i]:
                select_out_of_place.append(translate_query_namespace(eval(self.q2_select_expr[i].getZ3())), 
                                                                     self.reverse_mapping[1], self.q2_info.std_alias_to_info)

        return select_err, select_out_of_place #, select_missing


    def test_eval(self, f):
        print(eval(f))
        print(str(eval(f)))
        return str(eval(f)) 



def examine_queries(q1: str, q2: str):
    """ Given two query strings, invoke the entire process of hinting.
    ### Input
        q1: str, 
        q2: str, 
    ### Return
        (str, str)[], list of pair of repair sites and fixes
    """
    q1_info = QueryInfo(q1)
    q2_info = QueryInfo(q2)
    qm = MappingInfo(q1_info, q2_info)
    min_dist = math.inf
    qt = None
    for i in range(len(qm.table_mappings)):
        tp = QueryTest(q1_info, q2_info, qm.z3_var_lookup, qm.table_mappings[i], qm.table_mappings_reverse[i])
        if min_dist > tp.mapping_dist:
            qt = tp
            min_dist = qt.mapping_dist

    # now we have picked the right query mapping for test
    # Test WHERE

    # Test GROUP-BY

    # Test HAVING

    # Test SELECT

    return




