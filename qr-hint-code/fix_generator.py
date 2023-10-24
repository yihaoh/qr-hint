from z3 import *
import pandas as pd
from copy import deepcopy
from qm import QM
from new_qm import QM as newQM
from boolean_parse_tree import BNode



class FixGenerator:
    """ A class for generating fixes given two BNode formula, z3 variable mapping and repair sites.

    Attributes
    ----------
    target: BNode
    root: BNode
    z3_var: dict, table alias --> [z3 var instance, one for each attr in table]
    result: (BNode, z3 formula)[], repair sites and their corresponding fixes
    all_preds: string[], all semantically unique predicates
    all_preds_z3: z3 obj[], all semantically unique predicates, coupled with all_preds
    pred_equiv_map: dict, string --> int, map predicate string to index in all_preds s.t. predicate is semantically equivalent to all_preds[i]
    solver: a z3 solver

    Methods
    -------
    get_fixes()
    derive_fixes()
    minimize_target()
    evaluate_row()
    extract_preds()
    check_equiv()
    """

    def __init__(self, q1: BNode, q2: BNode, z3_var: dict, repair_sites: list):
        self.target = q1
        self.root = q2
        self.z3_var = z3_var
        self.rs = repair_sites
        self.result = None
        
        self.all_preds = []
        self.all_preds_z3 = []
        self.pred_equiv_map = {}

        self.solver = Solver()
        # tt = pd.DataFrame(columns=self.all_preds + ['lower', 'upper', 'final'], index=range(0, 2**(len(self.all_preds))))


    def get_fixes(self):
        if self.result == None:
            self.result = self.derive_fixes(self.root, eval(self.target.getZ3()), eval(self.target.getZ3()), self.rs)
        # print(len(self.all_preds))
        # print(self.all_preds)
        return self.result


    def derive_fixes(self, root: BNode, lower: BoolRef, upper: BoolRef, rs: list):
        """ Overall routine for deriving fixes INDIVIDUALLY.
        ### Input
            root: BNode, root of the entire wrong formula
            lower: z3 formula, lowerbound, usually called with the target formula
            upper: z3 formula, upperbound. usually called with the target formula
            rs: BNode[], a set of repair sites
        ### Return
            (BNode[], (z3 formula, fix_size))[], BNode are all repair sites at a level, z3 formula are the corresponding fix
        """
        if root.bounds[0] == root.bounds[1] or root.type != 'log':
            return []
        elif root.val == 'Not':
            return self.derive_fixes(root.children[0], Not(upper), Not(lower), rs)
        elif root == rs[0]:  # only valid when the entire tree is a repair site
            self.all_preds = []
            self.all_preds_z3 = []
            self.extract_preds(lower)
            self.extract_preds(upper)
            rs_target = self.minimize_target(lower, upper)
            res = [([root], rs_target)]
            return res
        
        # logical AND/OR, need to consider multiple repair sites
        rs_cur_level = []
        nrs_cur_level = []
        # split the child nodes
        for child in root.children:
            if child in rs:
                rs_cur_level.append(child)
            elif child.type in ['log', 'pred']:
                nrs_cur_level.append(child)

        # create new bounds for all nrs (treat all nrs as a separate child)
        nrs_repair_lower = 'True' if root.val == 'And' else 'False'
        nrs_repair_upper = 'True' if root.val == 'And' else 'False'
        for nrs in nrs_cur_level:
            # print(nrs.type, eval(nrs.getZ3()))
            nrs_repair_lower = f'{root.val}({nrs_repair_lower}, {nrs.bounds[0]})'
            nrs_repair_upper = f'{root.val}({nrs_repair_upper}, {nrs.bounds[1]})'


        # combine all rs and treat it as one rs and compute target bounds
        res = []
        if rs_cur_level:
            rs_target_lower = lower if root.val == 'And' else Or(And(lower, Not(eval(nrs_repair_lower))), False)
            rs_target_upper = And(Or(upper, Not(eval(nrs_repair_upper))), True) if root.val == 'And' else upper
            self.all_preds = []
            self.all_preds_z3 = []
            self.extract_preds(rs_target_lower)
            self.extract_preds(rs_target_upper)
            rs_target = self.minimize_target(rs_target_lower, rs_target_upper)
            res = [(rs_cur_level, rs_target)]

        # for each non-repair site, push down new target bounds
        for i in range(len(nrs_cur_level)):
            residual = nrs_cur_level[0:i] + nrs_cur_level[i + 1:]
            tp_repair_lower = False
            tp_repair_upper = True
            for n in residual:
                tp_repair_lower = And(tp_repair_lower, eval(n.bounds[0])) if root.val == 'And' else Or(tp_repair_lower, eval(n.bounds[0]))
                tp_repair_upper = And(tp_repair_upper, eval(n.bounds[1])) if root.val == 'And' else Or(tp_repair_upper, eval(n.bounds[1]))

            tp_target_lower = lower if root.val == 'And' else Or(And(lower, Not(tp_repair_lower)), eval(nrs_cur_level[i].bounds[0]))
            tp_target_upper = upper if root.val == 'Or' else And(Or(upper, Not(tp_repair_upper)), eval(nrs_cur_level[i].bounds[1]))
            res += self.derive_fixes(nrs_cur_level[i], tp_target_lower, tp_target_upper, rs)

        return res


    def minimize_target(self, lower: BoolRef, upper: BoolRef):
        """ Given lower and upper bounds, derive a minimum formula between them.
        ### Input
            lower: z3 obj, formula in z3 syntax tree
            upper: z3 obj, formula in z3 syntax tree
        ### Return
            (z3 obj, int), a minimum formula syntax tree and its size
        """
        tt = pd.DataFrame(columns=self.all_preds + ['lower', 'upper', 'final'], index=range(0, 2**(len(self.all_preds))))
        # print("table dimension: ", 2**(len(self.all_preds)), len(self.all_preds))

        index = 0
        end = 2**(len(self.all_preds))
        self.solver.reset()

        minterms, dontcares = [], []
        alias = [chr(ord('A') + x) for x in range(len(self.all_preds))]

        while index < end:
            self.solver.reset()
            tp_row = format(index, f'0{len(self.all_preds)}b')

            # assign truth value of variables
            for i in range(len(tp_row)):
                tt.at[index, self.all_preds[i]] = tp_row[i]

                if tp_row[i] == '1':
                    self.solver.add(self.all_preds_z3[i])
                else:
                    self.solver.add(Not(self.all_preds_z3[i]))

            if self.solver.check() == sat:
                # assign truth value for formulae
                tt.at[index, 'lower'] = self.evaluate_row(tp_row, lower)
                tt.at[index, 'upper'] = self.evaluate_row(tp_row, upper)
                tt.at[index, 'final'] = tt.at[index, 'upper'] if tt.at[index, 'lower'] == tt.at[index, 'upper'] else 'x'
                if tt.at[index, 'final'] == '1':
                    minterms.append(index)
                elif tt.at[index, 'final'] == 'x':
                    dontcares.append(index)
            else:
                tt.at[index, 'lower'] = 'x'
                tt.at[index, 'upper'] = 'x'
                tt.at[index, 'final'] = 'x'
                dontcares.append(index)
            index += 1

        # no minterm, false
        if not minterms:
            return eval('False'), 1

        # minimize formula
        # print(minterms, dontcares, alias)
        qm = QM(minterms, dontcares, alias)
        solss = sorted(qm.minimize(), key=len)
        sols = solss[0]

        # qm = newQM(alias)
        # sols = qm.get_function(qm.solve(minterms, dontcares)[1], 0)
        # print(sols)
        if sols == '1':
            return eval('True'), 1

        # parse min
        terms = [s.strip() for s in sols.split('+')]
        res = None
        sz = 0
        for term in terms:
            j = 0
            cur_clause = None
            while j < len(term):
                k = ord(term[j]) - ord('A')
                if j + 1 < len(term) and term[j+1] == '\'':
                    sz += 3 if cur_clause is not None else 2
                    cur_clause = And(cur_clause, Not(self.all_preds_z3[k])) if cur_clause is not None else Not(self.all_preds_z3[k])
                    j += 2
                else:
                    sz += 2 if cur_clause is not None else 1
                    cur_clause = And(cur_clause, self.all_preds_z3[k]) if cur_clause is not None else self.all_preds_z3[k]
                    j += 1
            sz += 1 if res is not None else 0        
            res = Or(res, cur_clause) if res is not None else cur_clause
        
        return res, sz


    def evaluate_row(self, pred_val: list, z3_tree: BoolRef) -> str:
        """ Given a z3 formula in its z3 obj syntax tree, evaluate its truth value based on an assignment.
        ### Input
            pred_val: string[], only contain 0 or 1, indicating the truth assginment of i-th predicate in all_preds
            z3_tree: z3 obj, syntax tree of z3 formula
        ### Return
            string, '1' if the truth value of z3_tree is evaluated to true, otherwise '0'
        """
        if z3_tree is None:
            return '1'

        cur_node = str(z3_tree.decl())

        # traverse down as usual]
        if cur_node == 'And':
            for c in z3_tree.children():
                tp = self.evaluate_row(pred_val, c)
                if tp == '0':
                    return '0'
            return '1'
        if cur_node == 'Or':
            for c in z3_tree.children():
                tp = self.evaluate_row(pred_val, c)
                if tp == '1':
                    return '1'
            return '0'
        elif cur_node == 'Not':
            res = self.evaluate_row(pred_val, z3_tree.children()[0])
            return '0' if res == '1' else '1'
        else:  # pred
            pred = str(z3_tree)
            if pred == 'True':
                return '1'
            elif pred == 'False':
                return '0'
            res = self.pred_equiv_map[pred]
            return pred_val[res]


    def extract_preds(self, z3_tree: BoolRef):
        """ Extract all semantically unique atomic predicates from z3 formula.
        ### Input
            z3_tree: z3 obj, root of the syntax tree of the formula
        ### Return
            none
        """
        if z3_tree is None:
            return
        
        cur_node = str(z3_tree.decl())
        if cur_node in ['And', 'Or']:
            for c in z3_tree.children():
                self.extract_preds(c)
        elif cur_node == 'Not':
            self.extract_preds(z3_tree.children()[0])
        else:  # pred
            pred = str(z3_tree)
            if pred in ['True', 'False']:
                return
            for i, p in enumerate(self.all_preds_z3):
                # print(p, z3_tree)
                if self.check_equiv(p, z3_tree):
                    self.pred_equiv_map[pred] = i
                    return
            
            self.all_preds.append(pred)
            self.all_preds_z3.append(z3_tree)
            self.pred_equiv_map[pred] = len(self.all_preds) - 1

        
    def check_equiv(self, p1: BoolRef, p2: BoolRef):
        """ Check the equivalence of two predicates
        ### Input
            p1: z3 obj, represent a predicate
            p2: z3 obj, represent a predicate

        ### Return
            bool, true if p1 is equivalent to p2, else false
        """
        tp_solver = Solver()
        tp_solver.add(Not(Implies(p1, p2)))
        if tp_solver.check() == sat:
            return False
        
        tp_solver.reset()
        tp_solver.add(Not(Implies(p2, p1)))
        if tp_solver.check() == sat:
            return False
        
        return True


    