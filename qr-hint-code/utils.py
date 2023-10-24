import re
from boolean_parse_tree import BNode
from copy import deepcopy
from z3 import *





def calculate_normalized_distance(q1: BoolRef, q2: BoolRef):
    """ Given two formula, calculate their normalized edit distance based on join conditions.
    ### Input
        q1: z3 obj
        q2: z3 obj
    ### Return
        int, the levenshtein edit distance between two string
    """

    def get_all_predicates(root: BoolRef):
        if root is None:
            return []
        cur_node = str(root.decl())
        if cur_node in ['And', 'Or']:
            res = []
            for c in root.children():
                res += get_all_predicates(c)
            return res
        elif cur_node == 'Not':
            return get_all_predicates(root.children()[0])
        else:  # pred
            return [deepcopy(root)]

    def normalize_predicate(pred: BoolRef):
        if pred is None:
            return ''

        op = str(pred.decl())
        op = '!=' if op == 'Distinct' else op
        left, right = [s for s in str(pred.children()[0]).split(' ') if '.' in s], [
            s for s in str(pred.children()[1]).split(' ') if '.' in s]

        # not a join condition, return empty string
        if not left or not right:
            return ''

        left.sort()
        right.sort()

        left_str, right_str = ' '.join(left), ' '.join(right)

        if op == '==':
            # rearrange with lexi order
            return f'{left_str} == {right_str}' if left_str > right_str else f'{right_str} == {left_str}'
        elif op == '>':
            # convert to <
            return f'{right_str} < {left_str}'
        elif op == '>=':
            # convert to <=
            return f'{right_str} <= {left_str}'
        elif op == '!=':
            return f'{left_str} != {right_str}' if left_str > right_str else f'{right_str} != {left_str}'
        return f'{left} {op} {right_str}'
        
    def edit_distance(s: str, t: str) -> int:
        n = len(s)
        m = len(t)
 
        prev = [j for j in range(m+1)]
        curr = [0] * (m+1)
 
        for i in range(1, n+1):
            curr[0] = i
            for j in range(1, m+1):
                if s[i-1] == t[j-1]:
                    curr[j] = prev[j-1]
                else:
                    mn = min(1 + prev[j], 1 + curr[j-1])
                    curr[j] = min(mn, 1 + prev[j-1])
            prev = curr.copy()
        return prev[m]

    p1, p2 = get_all_predicates(q1), get_all_predicates(q2)
    res1, res2 = [normalize_predicate(p) for p in p1], [
        normalize_predicate(p) for p in p2]
    res1.sort()
    res2.sort()
    s1, s2 = ' '.join(res1), ' '.join(res2)
    return edit_distance(s1, s2)


def convert_to_nary_tree(root: BNode):
    """ Convert a binary syntax tree to n-ary syntax tree
    ### Input
        root: BNode, root of the binary syntax tree to be converted

    ### Return
        BNode, new root of the n-ary syntax tree
    """
    if root.type == 'pred':
        return root

    if root.type == 'log' and root.val in ['And', 'Or']:
        left = convert_to_nary_tree(root.children[0])
        right = convert_to_nary_tree(root.children[1])

        # collapse if children are same logical operator
        if root.type == left.type and root.val == left.val and root.type == right.type and root.val == right.val:
            root.children = left.children + right.children
        elif root.type == left.type and root.val == left.val:
            root.children = left.children + [root.children[1]]
        elif root.type == right.type and root.val == right.val:
            root.children = [root.children[0]] + right.children

        # reset children's parent to root
        for n in root.children:
            n.parent = root

    # reset z3_formula and size
    root.z3_formula = None
    root.size = 0
    return root


def convert_binary_to_cnf(root: BNode):
    """ Convert a binary syntax tree of a formula to an a-nary tree that represents the CNF of the formula.
    ### Input
        root: BNode, root of the binary syntax tree
    ### Return
        BNode, new root for the n-ary CNF tree
    """
    if root.type == 'pred':
        return deepcopy(root)

    elif root.type == 'log' and root.val == 'And':
        # if And, simply connect the left and right to maintain CNF at current node
        left = convert_to_nary_tree(convert_binary_to_cnf(root.children[0]))
        right = convert_to_nary_tree(convert_binary_to_cnf(root.children[1]))
        new_root = BNode(root.val, root.type, root.select_xid)
        new_root.children = [left, right]
        return convert_to_nary_tree(new_root)

    elif root.type == 'log' and root.val == 'Or':
        left = convert_to_nary_tree(convert_binary_to_cnf(root.children[0]))
        right = convert_to_nary_tree(convert_binary_to_cnf(root.children[1]))

        # combine clauses from both left and right to make new CNF
        left_clauses = [left] if left.type != 'Or' else left.children
        right_clauses = [right] if right.type != 'Or' else right.children
        new_root = BNode('And', 'log', root.select_xid)
        for lc in left_clauses:
            for rc in right_clauses:
                tp = BNode('Or', 'log', root.select_xid)
                tp.children = [deepcopy(lc), deepcopy(rc)]
                new_root.children.append(tp)
        return convert_to_nary_tree(new_root)

    elif root.type == 'log' and root.val == 'Not':
        raise NotImplementedError(
            'QueryTest: convert_binary_to_cnf does not support NOT')

    return None


def get_non_agg_cnf_clauses(root: BNode):
    """ Deep copy non-aggregate cnf clauses.
    ### Input
        root: BNode, root of the cnf formula
    ### Return
        BNode if root of the non-agg clauses, else None
    """
    if root.type == 'pred':
        return deepcopy(root) if re.search(r'COUNT|AVG|SUM|MAX|MIN|count|avg|sum|max|min', root.getZ3()) == None else None

    if root.type == 'log' and root.val == 'And':
        new_root = BNode(root.val, root.type, root.select_xid)
        for c in root.children:
            if c.type == 'log' and c.val == 'Or':
                is_agg_clause = False
                for s in c.children:
                    if re.search(r'COUNT|AVG|SUM|MAX|MIN|count|avg|sum|max|min', s.getZ3()) != None:
                        is_agg_clause = True
                        break
                if not is_agg_clause:
                    new_root.children.append(deepcopy(c))

            elif c.type == 'pred' and re.search(r'COUNT|AVG|SUM|MAX|MIN|count|avg|sum|max|min', c.getZ3()) == None:
                new_root.children.append(deepcopy(c))
        return new_root if new_root.children else None

    return None


def lca_nary(root: BNode, lower: str, upper: str):
    if not root:
        return (None, '', '')

    # single repair site?
    if root.bounds[0] == 'False' and root.bounds[1] == 'True':
        return root, lower, upper

    if len(root.children) == 1:
        new_lower = f'Not({upper})'
        new_upper = f'Not({lower})'
        return lca_nary(root.children[0], new_lower, new_upper)

    # split repair sites and non-repair sites
    rs_cur_level = []
    nrs_cur_level = []
    for child in root.children:
        if child.bounds[0] == child.bounds[1]:
            nrs_cur_level.append(child)
        elif child.bounds[0] != child.bounds[1]:
            rs_cur_level.append(child)

    # if multiple children have inequal bounds, then this is LCA
    if len(rs_cur_level) > 1:
        return (root, lower, upper)

    # otherwise only 1 child has inequal bounds, traverse down
    nrs_repair_lower = 'True' if root.val == 'And' else 'False'
    nrs_repair_upper = 'True' if root.val == 'And' else 'False'
    for nrs in nrs_cur_level:
        # print(nrs.type, eval(nrs.getZ3()))
        nrs_repair_lower = f'{root.val}({nrs_repair_lower}, {nrs.bounds[0]})'
        nrs_repair_upper = f'{root.val}({nrs_repair_upper}, {nrs.bounds[1]})'

    rs_target_lower = lower if root.val == 'And' else f'Or(And({lower}, Not({nrs_repair_lower})), {rs_cur_level[0].bounds[0]})'
    rs_target_upper = f'And(Or({upper}, Not({nrs_repair_upper})), {rs_cur_level[0].bounds[1]})' if root.val == 'And' else upper
    return lca_nary(rs_cur_level[0], rs_target_lower, rs_target_upper)


def lca(root: BNode, lower: str, upper: str):
    if not root:
        return None, '', ''

    # single repair site?
    if root.bounds[0] == 'False' and root.bounds[1] == 'True':
        return root, lower, upper

    if len(root.children) == 2:
        if root.children[0].bounds[0] == root.children[0].bounds[1]:
            if root.val == 'And':
                new_lower = f'Or({lower}, {root.children[1].bounds[0]})'
                new_upper = f'And(Or({upper}, Or({root.children[1].bounds[0]}, Not({root.children[0].bounds[1]}))), {root.children[1].bounds[1]})'
            elif root.val == 'Or':
                new_lower = f'Or(And({lower}, And({root.children[1].bounds[1]}, Not({root.children[0].bounds[0]}))), {root.children[1].bounds[0]})'
                new_upper = f'And({upper}, {root.children[1].bounds[1]})'
            return lca(root.children[1], new_lower, new_upper)
        elif root.children[1].bounds[0] == root.children[1].bounds[1]:
            if root.val == 'And':
                new_lower = f'Or({lower}, {root.children[0].bounds[0]})'
                new_upper = f'And(Or({upper}, Or({root.children[0].bounds[0]}, Not({root.children[1].bounds[1]}))), {root.children[0].bounds[1]})'
            elif root.val == 'Or':
                new_lower = f'Or(And({lower}, And({root.children[0].bounds[1]}, Not({root.children[1].bounds[0]}))), {root.children[0].bounds[0]})'
                new_upper = f'And({upper}, {root.children[0].bounds[1]})'
            return lca(root.children[0], new_lower, new_upper)
        else:  # reach lca, return
            return root, lower, upper
    elif len(root.children) == 1:
        new_lower = f'Not({upper})'
        new_upper = f'Not({lower})'
        return lca(root.children[0], new_lower, new_upper)

    # root has no child
    return root, lower, upper

# def lca(root, nodes):
#     if root in nodes:
#         return root, 1

#     if len(root.children) == 2:
#         left, left_cnt = lca(root.children[0])
#         right, right_cnt = lca(root.children[1])
#         if left_cnt and right_cnt:
#             return root, left_cnt + right_cnt
#         elif left_cnt:
#             return left, left_cnt
#         elif right_cnt:
#             return right, right_cnt
#         else:
#             return None, 0
#     elif len(root.children) == 1:
#         return lca(root.children[0])

#     # no children, leaf node but not in nodes
#     return None, 0


def translate_query_namespace(q: BoolRef, mapping: dict, std_alias_to_info: dict):
    token = str(q.decl())
    token = '!=' if token == 'Distinct' else token
    if token in ['And', 'Or']:
        res = []
        for c in q.children():
            res.append(translate_query_namespace(c, mapping, std_alias_to_info))
        return '(' + f' {token} '.join(res) + ')'
    if token in ['+', '-', '*', '/', '==', '<', '>', '<=', '>=', '!=']:   # Distcint -> !=
        return f'{translate_query_namespace(q.children()[0], mapping, std_alias_to_info)} {token} {translate_query_namespace(q.children()[1], mapping, std_alias_to_info)}'
    if token in ['Not', 'COUNT', 'MAX', 'AVG', 'MIN', 'SUM']:
        return f'{token}({translate_query_namespace(q.children[0], mapping, std_alias_to_info)})'
    
    # constant
    if token in ['String', 'Int', 'Real']:
        return str(q)
    
    # variables
    tp = token.split('.')
    return f'{std_alias_to_info[mapping[tp[0]]][2]}.{tp[1]}'


def get_leaf_nodes(syn_tree: BNode):
    # get all leaf nodes in left-to-right order
    if not syn_tree.children:
        return [syn_tree]

    if len(syn_tree.children) == 1:
        return get_leaf_nodes(syn_tree.children[0])

    left = get_leaf_nodes(syn_tree.children[0])
    right = get_leaf_nodes(syn_tree.children[1])

    return left + right


def repair_cost(p1: BNode, p2: BNode, repair: list) -> float:
    res = 0
    p1_size, p2_size = p1.get_size(), p2.get_size()
    for r in repair:
        res += sum([r[0][i].get_size() for i in range(len(r[0]))]) + r[1][1]
    res /= p1_size + p2_size
    res += 1 / 6 * len(repair)
    return round(res, 2)


def is_conjunctive(n: BNode):
    if n.type != 'log' or n.val != 'And':
        return False
    for c in n.children:
        if c.type != 'pred':
            return False
    return True

def get_actual_rs_count(rs: list):
    tp = {}
    for r in rs:
        if r.parent == None:
            tp['none'] = 1
            continue
        if r.parent.nid in tp:
            continue
        tp[r.parent.nid] = 1
    return len(tp)
