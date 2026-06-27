import re
from .boolean_parse_tree import BNode
from copy import deepcopy
from z3 import *


# Date literal cache: maps integer string to DATE literal string
# e.g., "19950315" → "DATE '1995-03-15'"
_date_literal_cache = {}


def date_literal_to_int(date_sql: str) -> str:
    """Convert DATE 'YYYY-MM-DD' to integer string 'YYYYMMDD' and cache the mapping.
    YYYYMMDD integers preserve date ordering, so <, >, =, >=, <= all work correctly.
    """
    date_str = date_sql[6:-1]  # Extract YYYY-MM-DD from DATE 'YYYY-MM-DD'
    date_int_str = date_str.replace("-", "")
    _date_literal_cache[date_int_str] = date_sql
    return date_int_str


def ast_to_str(clauses, num_var):
    """
    ### Input
        clauses: pyeda expr obj
        num_var: int
    ### Return
        str, DNF in form ABC + A'B...
    """
    # DEBUG: Check if clauses is an int (happens when formula is constant)
    if isinstance(clauses, int):
        return str(clauses)

    if clauses[0] == "const":
        return str(clauses[1])

    if clauses[0] == "lit":
        res = chr(ord("A") + abs(abs(clauses[1]) - num_var))
        if clauses[1] < 0:
            res += "'"
        return res

    res = ""
    if clauses[0] == "and":
        for i in range(1, len(clauses)):
            cur_v = clauses[i]
            # Handle case where cur_v is an int (constant)
            if isinstance(cur_v, int):
                continue
            if cur_v[0] == "lit":
                res += chr(ord("A") + abs(abs(cur_v[1]) - num_var))
                if cur_v[1] < 0:
                    res += "'"
        return res

    # clause start with 'or'
    for i in range(1, len(clauses)):
        cur_and = clauses[i]
        # Handle case where cur_and is an int (constant)
        if isinstance(cur_and, int):
            continue
        for j in range(1, len(cur_and)):
            cur_v = cur_and[j]
            # Handle case where cur_v is an int (constant)
            if isinstance(cur_v, int):
                continue
            if cur_v[0] == "lit":
                res += chr(ord("A") + abs(abs(cur_v[1]) - num_var))
                if cur_v[1] < 0:
                    res += "'"
        if i < len(clauses) - 1:
            res += "+"
    return res


def convert_to_nary_tree(root: BNode):
    """Convert a binary syntax tree to n-ary syntax tree
    ### Input
        root: BNode, root of the binary syntax tree to be converted

    ### Return
        BNode, new root of the n-ary syntax tree
    """
    if not root:
        return None

    if root.type == "pred":
        return root

    if root.type == "log" and root.val in ["And", "Or"]:
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


def lca_nary(root: BNode, lower: str, upper: str):
    if not root:
        return (None, "", "")

    # single repair site?
    if root.bounds[0] == "False" and root.bounds[1] == "True":
        return root, lower, upper

    if len(root.children) == 1:
        new_lower = f"Not({upper})"
        new_upper = f"Not({lower})"
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
    nrs_repair_lower = "True" if root.val == "And" else "False"
    nrs_repair_upper = "True" if root.val == "And" else "False"
    for nrs in nrs_cur_level:
        # print(nrs.type, eval(nrs.getZ3()))
        nrs_repair_lower = f"{root.val}({nrs_repair_lower}, {nrs.bounds[0]})"
        nrs_repair_upper = f"{root.val}({nrs_repair_upper}, {nrs.bounds[1]})"

    rs_target_lower = (
        lower if root.val == "And" else f"Or(And({lower}, Not({nrs_repair_lower})), {rs_cur_level[0].bounds[0]})"
    )
    rs_target_upper = (
        f"And(Or({upper}, Not({nrs_repair_upper})), {rs_cur_level[0].bounds[1]})" if root.val == "And" else upper
    )
    return lca_nary(rs_cur_level[0], rs_target_lower, rs_target_upper)


def lca(root: BNode, lower: str, upper: str):
    if not root:
        return None, "", ""

    # single repair site?
    if root.bounds[0] == "False" and root.bounds[1] == "True":
        return root, lower, upper

    if len(root.children) == 2:
        if root.children[0].bounds[0] == root.children[0].bounds[1]:
            if root.val == "And":
                new_lower = f"Or({lower}, {root.children[1].bounds[0]})"
                new_upper = f"And(Or({upper}, Or({root.children[1].bounds[0]}, Not({root.children[0].bounds[1]}))), {root.children[1].bounds[1]})"
            elif root.val == "Or":
                new_lower = f"Or(And({lower}, And({root.children[1].bounds[1]}, Not({root.children[0].bounds[0]}))), {root.children[1].bounds[0]})"
                new_upper = f"And({upper}, {root.children[1].bounds[1]})"
            return lca(root.children[1], new_lower, new_upper)
        elif root.children[1].bounds[0] == root.children[1].bounds[1]:
            if root.val == "And":
                new_lower = f"Or({lower}, {root.children[0].bounds[0]})"
                new_upper = f"And(Or({upper}, Or({root.children[0].bounds[0]}, Not({root.children[1].bounds[1]}))), {root.children[0].bounds[1]})"
            elif root.val == "Or":
                new_lower = f"Or(And({lower}, And({root.children[0].bounds[1]}, Not({root.children[1].bounds[0]}))), {root.children[0].bounds[0]})"
                new_upper = f"And({upper}, {root.children[0].bounds[1]})"
            return lca(root.children[0], new_lower, new_upper)
        else:  # reach lca, return
            return root, lower, upper
    elif len(root.children) == 1:
        new_lower = f"Not({upper})"
        new_upper = f"Not({lower})"
        return lca(root.children[0], new_lower, new_upper)

    # root has no child
    return root, lower, upper


def translate_query_namespace(q: BoolRef, mapping: dict, std_alias_to_info: dict):
    # Handle Python bool values (returned by minimize_target when fix is True/False)
    if isinstance(q, bool):
        return "True" if q else "False"

    if q is None:
        return None

    token = str(q.decl())
    token = "!=" if token == "Distinct" else token
    if token in ["And", "Or"]:
        res = []
        for c in q.children():
            res.append(translate_query_namespace(c, mapping, std_alias_to_info))
        return "(" + f" {token} ".join(res) + ")"
    if token in ["+", "-", "*", "/", "==", "<", ">", "<=", ">=", "!="]:  # Distcint -> !=
        return f"{translate_query_namespace(q.children()[0], mapping, std_alias_to_info)} {token} {translate_query_namespace(q.children()[1], mapping, std_alias_to_info)}"
    if token in ["Not", "COUNT", "MAX", "AVG", "MIN", "SUM", "COUNTINT", "COUNTSTR"]:
        return f"{token}({translate_query_namespace(q.children()[0], mapping, std_alias_to_info)})"

    # constant
    if token in ["String", "Int", "Real"]:
        val_str = str(q)
        return _date_literal_cache.get(val_str, val_str)

    # numeric constant (Z3 numerals: decl() returns the value as string)
    val_str = str(q)
    if val_str in _date_literal_cache:
        return _date_literal_cache[val_str]

    # variables
    tp = token.split(".")
    return f"{std_alias_to_info[mapping[tp[0]]][2]}.{tp[1]}"


def repair_cost(p1: BNode, p2: BNode, repair: list) -> float:
    res = 0
    p1_size, p2_size = p1.get_size(), p2.get_size()
    for r in repair:
        res += sum([r[0][i].get_size() for i in range(len(r[0]))]) + r[1][1]
    res /= p1_size + p2_size
    res += 1 / 6 * len(repair)
    return round(res, 2)


def is_conjunctive(n: BNode):
    if n.type == "log" and n.val != "And":
        return False
    for c in n.children:
        if c.type != "pred":
            return False
    return True


def get_actual_rs_count(rs: list[BNode]):
    tp = {}
    for r in rs:
        if r.parent == None:
            tp["none"] = 1
            continue
        if r.parent.nid in tp:
            continue
        tp[r.parent.nid] = 1
    return len(tp)
