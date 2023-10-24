from z3 import *
import pandas as pd
from copy import deepcopy
from qm import QM
from new_qm import QM as newQM
from boolean_parse_tree import BNode



class FixOptimizer:
    """
    Class FixOptimizer built for Hinting project.
    The main truth table contain truth value for lowerbound, upperbound, target formula given lower/upper
    and target formula contains all repair sites.

    Attributes
    ----------
    lower: str
        z3-ready formula for lowerbound at LCA of repair sites
    upper: str
        z3-ready formula for upperbound at LCA of repair sites
    rs_lca: BNode
        lowest common ancestor of all repair sites
    rs: BNode[]
        a list of repair sites in rs_lca
    rs_str: string []
        a list of BNode nid used as repair site identifiers
    all_preds: str[]
        a list of all syntactically unique predicates from lower, upper and eval(rs_lca)
    all_preds_z3: BoolRef[]
        a list of all syntactically unique predicates from lower, upper and eval(rs_lca)
    rs_dedup_map: dict, nid -> nid
        the value nid and the key nid share the same parents, thus being considered as a single repair site
    main: DataFrame
        consistency table for lower, upper, minimum bounded formula and target_lca
    constraint: DataFrame
        constraint table for repair sites with rs_str + all_preds

    Methods
    -------
    
    """

    def __init__(self, lower: BoolRef, upper: BoolRef, rs_lca: BNode, repair_sites: list, z3_var: dict):
        self.lower = lower
        self.upper = upper
        self.rs_lca = rs_lca
        self.rs = repair_sites  # rs in rs_lca
        self.z3_var = z3_var
        
        self.all_preds = []
        self.all_preds_z3 = []
        self.fixes = []
        self.solver = Solver()
        self.rs_lookup = { x.nid: x for x in self.rs }
        
        self.rs_str = []
        self.rs_dedup_map = self.dedup_repair_site(self.rs_lca)

        self.extract_preds(lower)
        self.extract_preds(upper)
        self.extract_preds(eval(self.rs_lca.getZ3()))

        # print(len(self.rs_str), len(self.all_preds))
        # print(self.all_preds)

        self.consistency = pd.DataFrame(columns=self.rs_str + self.all_preds + ['lower', 'upper', 'bound', 'target_lca'],
                                index=range(0, 2**(len(self.all_preds) + len(self.rs_str))))

        self.constraint = pd.DataFrame(columns=self.all_preds + ['assignments'] + self.rs_str,
                                        index=range(0, 2**len(self.all_preds)))

        self.fill_consistency_table()
        self.fill_constraint_table()

    def get_fixes(self):
        return self.fixes


    def dedup_repair_site(self, root: BNode) -> dict:
        if root.type == 'pred':
            if root in self.rs:
                self.rs_str.append(root.nid)
            return {}
        cur_level_rs = []
        cur_res = {}
        for c in root.children:
            if c in self.rs:
                cur_level_rs.append(c)
            else:
                cur_res.update(self.dedup_repair_site(c))
        if cur_level_rs:
            self.rs_str.append(cur_level_rs[0].nid)
        for i in range(1, len(cur_level_rs)):
            cur_res[cur_level_rs[i].nid] = cur_level_rs[0].nid
        return cur_res


    def get_qm_result(self, minterms, dontcares, alias):
        # print(minterms, dontcares, alias)
        qm = QM(minterms, dontcares, alias)
        sols = sorted(qm.minimize(), key=len)[0]
        # print(sols)

        # qm = newQM(alias)
        # sols = qm.get_function(qm.solve(minterms, dontcares)[1], 2)
        # print(sols)

        if sols == '1':
            return 'True'
        return sols


    def fill_constraint_table(self):
        index = 0
        end = 2 ** len(self.all_preds)
        all_assignments = [format(x, f'0{len(self.rs_str)}b') for x in range(2**len(self.rs_str))]

        # first fill in each row except for the truth value of fixes
        while index < end:
            tp_row = format(index, f'0{len(self.all_preds)}b')

            # assign truth value of variables
            for i in range(len(tp_row)):
                self.constraint.at[index, self.all_preds[i]] = tp_row[i]

            row_assignments = deepcopy(all_assignments)
            # for each combinations of truth value, look for possibilities
            for i in range(2**len(self.rs_str)):
                cur_assignment = format(i, f'0{len(self.rs_str)}b')
                bound = self.consistency.at[index + i * 2 ** len(self.all_preds), 'bound']
                target_lca = self.consistency.at[index + i * 2 ** len(self.all_preds), 'target_lca']
                if bound != target_lca and bound != 'x':
                    if cur_assignment in row_assignments:
                        row_assignments.remove(cur_assignment)
            
            self.constraint.at[index, 'assignments'] = row_assignments

            index += 1
        
        # now greedily fill in the truth value for fixes
        alias = [chr(ord('A') + x) for x in range(len(self.all_preds))]
        for i, s in enumerate(self.rs_str):
            minterms, dontcares = [], []
            for j in range(end):
                prev = 'x' if i == 0 else self.constraint.at[j, self.rs_str[i-1]]
                assignments = [x for x in self.constraint.at[j, 'assignments'] if x[i-1] == prev] if prev != 'x' else self.constraint.at[j, 'assignments']
                self.constraint.at[j, self.rs_str[i]] = '0'
                has0, has1 = False, False
                for k in range(len(assignments)):
                    if assignments[k][i] == '0':
                        has0 = True
                    else:
                        has1 = True
                if has0 and has1:
                    dontcares.append(j)
                elif has1:
                    minterms.append(j)

            raw_fix = self.get_qm_result(minterms, dontcares, alias) if minterms else 'False'
            tp_rs = [self.rs_lookup[s]] + [self.rs_lookup[x] for x,y in self.rs_dedup_map.items() if y == s]
            terms = [s.strip() for s in raw_fix.split('+')]
            if raw_fix == 'True':
                self.constraint[s] = '1'
                self.fixes.append((tp_rs, (eval('True'), 1)))
            elif raw_fix == 'False':
                self.constraint[s] = '0'
                self.fixes.append((tp_rs, (eval('False'), 1)))
            else:
                # fill selective rows with one, all other are zero
                for t in terms:
                    j = 0
                    filterData = self.constraint
                    while j < len(t):
                        k = ord(t[j]) - ord('A')
                        if j + 1 < len(t) and t[j+1] == '\'':
                            filterData = filterData[filterData[self.all_preds[k]] == '0']
                            j += 2
                        else:
                            filterData = filterData[filterData[self.all_preds[k]] == '1']
                            j += 1
                    for idx in list(filterData.index.values):
                        self.constraint.at[idx, self.rs_str[i]] = '1'
        
                res = []
                sz = 0
                for term in terms:
                    j = 0
                    cur_clause = []
                    while j < len(term):
                        k = ord(term[j]) - ord('A')
                        if j + 1 < len(term) and term[j+1] == '\'':
                            cur_clause.append(Not(self.all_preds_z3[k]))
                            j += 2
                            sz += 2
                        else:
                            cur_clause.append(self.all_preds_z3[k])
                            j += 1
                            sz += 1
                    if len(cur_clause) > 1:
                        res.append(And(cur_clause))
                        sz += 1
                    else:
                        res.append(cur_clause[0])
                    
                if len(res) > 1:
                    sz += 1
                    self.fixes.append((tp_rs, (Or(res), sz)))
                else:
                    self.fixes.append((tp_rs, (res[0], sz)))
                
        
    def fill_consistency_table(self):
        index = 0
        end = 2**(len(self.all_preds) + len(self.rs_str))
        self.solver.reset()

        while index < end:
            self.solver.reset()
            tp_row = format(index, f'0{len(self.rs_str) + len(self.all_preds)}b')

            # assign truth value of variables
            for i in range(len(tp_row)):
                if i < len(self.rs_str):
                    self.consistency.at[index, self.rs_str[i]] = tp_row[i]
                else:
                    self.consistency.at[index, self.all_preds[i - len(self.rs_str)]] = tp_row[i]

                    if tp_row[i] == '1':
                        self.solver.add(self.all_preds_z3[i - len(self.rs_str)])
                    else:
                        self.solver.add(Not(self.all_preds_z3[i - len(self.rs_str)]))

            if self.solver.check() == sat:
                # assign truth value for formulae
                self.consistency.at[index, 'lower'] = self.evaluate_row_z3(tp_row, self.lower)
                self.consistency.at[index, 'upper'] = self.evaluate_row_z3(tp_row, self.upper)
                self.consistency.at[index, 'bound'] = self.consistency.at[index, 'upper'] if self.consistency.at[index, 'lower'] == self.consistency.at[index, 'upper'] else 'x'
                self.consistency.at[index, 'target_lca'] = self.evaluate_row_bnode(tp_row, self.rs_lca)
            else:
                self.consistency.at[index, 'lower'] = 'x'
                self.consistency.at[index, 'upper'] = 'x'
                self.consistency.at[index, 'bound'] = 'x'
                self.consistency.at[index, 'target_lca'] = 'x'
            index += 1
            # row_num += 1


    def extract_preds(self, z3_tree):
        if z3_tree is None:
            return
        
        cur_node = str(z3_tree.decl())
        if cur_node in ['And', 'Or']:
            for c in z3_tree.children():
                self.extract_preds(c)
        elif cur_node == 'Not':
            self.extract_preds(z3_tree.children()[0])
        else:  # pred
            # if not self.implies(self.preds_conj, z3_tree):
            #     pred = str(z3_tree)
            #     self.all_preds.append(pred)
            #     self.all_preds_z3.append(z3_tree)
            #     self.preds_conj = And(z3_tree, self.preds_conj)
            pred = str(z3_tree)
            if pred in ['True', 'False']:
                return
            if pred not in self.all_preds:
                self.all_preds.append(pred)
                self.all_preds_z3.append(z3_tree)

                    
    def implies(self, a, b):
        self.solver.reset()
        self.solver.add(Not(Implies(a, b)))
        res = self.solver.check()
        return res == unsat


    def evaluate_row_bnode(self, pred_val: list, root: BNode):
        if root is None:
            return '1'
        
        if root.nid in self.rs_dedup_map:
            return pred_val[self.rs_str.index(self.rs_dedup_map[root.nid])]
        
        if root.nid in self.rs_str:
            return pred_val[self.rs_str.index(root.nid)]
        
        if root.val == 'And':
            for c in root.children:
                if self.evaluate_row_bnode(pred_val, c) == '0':
                    return '0'
            return '1'
        elif root.val == 'Or':
            for c in root.children:
                if self.evaluate_row_bnode(pred_val, c) == '1':
                    return '1'
            return '0'
        elif root.val == 'Not':
            res = self.evaluate_row_bnode(pred_val, root.children[0])
            return '0' if res == '1' else '1'
        # pred
        res = self.all_preds.index(str(eval(root.getZ3())))
        return pred_val[len(self.rs_str) + res]


    def evaluate_row_z3(self, pred_val: list, z3_tree: BoolRef):
        if z3_tree is None:
            return '1'

        cur_node = str(z3_tree.decl())

        # otherwise traverse down as usual
        if cur_node == 'And':
            for c in z3_tree.children():
                tp = self.evaluate_row_z3(pred_val, c)
                if tp == '0':
                    return '0'
            return '1'
        if cur_node == 'Or':
            for c in z3_tree.children():
                tp = self.evaluate_row_z3(pred_val, c)
                if tp == '1':
                    return '1'
            return '0'
        elif cur_node == 'Not':
            res = self.evaluate_row_z3(pred_val, z3_tree.children()[0])
            return '0' if res == '1' else '1'
        else:  # pred
            pred = str(z3_tree)
            if pred == 'True':
                return '1'
            elif pred == 'False':
                return '0'
            res = self.all_preds.index(pred)
            return pred_val[len(self.rs_str) + res]
    
