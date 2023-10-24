import uuid


class BNode:
    """
    A class used to define the nodes of syntax tree of Boolean formula.
    Note that this BNode is used both for binary tree and n-ary tree.

    Attributes
    ----------
    val: string
        one of the following: logical/arithmetic/string/comparison operator, const or var
    type: string
        one of the following: log, pred, expr, var, const
    children: list of BNode
        child nodes of the current node, usually have size 1 or 2
    z3_formula: string
        z3 formula represents the entire subtree rooted at the node, can be passed to eval() directly
    size: int
        the size of subtree
    bounds: (string, string)
        lowerbound and upperbound in z3-eval-ready format, update dynamically
    parent: BNode
        the parent node of the current node
    select_xid: string
        the xid of the root of this SELECT context

    Methods
    -------
    getZ3()
    get_size()
    """

    def __init__(self, val, type, select_xid=None, parent=None):
        self.nid = str(uuid.uuid4())
        self.val = val
        self.type = type        # log, pred, expr, var, const
        self.children = []
        self.z3_formula = None
        self.size = 0
        self.bounds = None
        self.parent = parent
        self.select_xid = select_xid


    def getZ3(self):
        """Return z3-ready formula for the subtree.
        ### Input: none
        ### Return: 
            string: z3 formula for the current subtree, ready for eval()         
        """

        # if z3_formula exists
        if self.z3_formula:
            return self.z3_formula

        if self.children:
            if self.type == 'log':
                if self.val == 'Not':
                    self.z3_formula = f'Not({self.children[0].getZ3()})'
                elif self.val in ['And', 'Or']:
                    self.z3_formula = f'{self.val}({self.children[0].getZ3()}, {self.children[1].getZ3()})'
                    for i in range(2, len(self.children)):
                        self.z3_formula = f'{self.val}({self.z3_formula}, {self.children[i]})'
                else:
                    raise NotImplementedError(f'{self.val} is not a supported logical operator.')

            elif self.type == 'pred':
                # comparison: <, >, <=, >=, =, <>, like
                if self.val == 'like':
                    # assuming left is expr or var, and right is always const
                    left, right = self.children[0], self.children[1]
                    arr = right.getZ3().split('%')
                    if len(arr) == 0:  
                        # no %, same as ==
                        self.z3_formula = f'{self.children[0].getZ3()} == {self.children[1].getZ3()}'
                    elif len(arr[0]) == 0 and len(arr[-1]) == 0:
                        # %xxx%
                        self.z3_formula = f'Contains({left.getZ3()}, {right.getZ3()})'
                    elif len(arr[0]) == 0:
                        # %xxx
                        self.z3_formula = f'SuffixOf({left.getZ3()}, {right.getZ3()})'
                    elif len(arr[-1]) == 0:
                        # xxx%
                        self.z3_formula = f'PrefixOf({left.getZ3()}, {right.getZ3()})'
                        pass
                    else:
                        raise NotImplementedError(f'like {right.getZ3()} is not supported.')
                elif self.val == '=':
                    self.z3_formula = f'{self.children[0].getZ3()} == {self.children[1].getZ3()}'
                elif self.val == '<>':
                    self.z3_formula = f'{self.children[0].getZ3()} != {self.children[1].getZ3()}'
                elif self.val in ['<', '>', '<=', '>=']:
                    self.z3_formula = f'{self.children[0].getZ3()} {self.val} {self.children[1].getZ3()}'
                else:
                    raise NotImplementedError(f'{self.val} is not a supported comparison operator.')

            elif self.type == 'expr':
                # expr: +, -, *, /, ||, AVG, SUM, COUNT, MAX, MIN
                if self.val in ['+', '-', '*', '/']:
                    self.z3_formula = f'{self.children[0].getZ3()} {self.val} {self.children[1].getZ3()}'
                elif self.val == '||':
                    self.z3_formula = f'{self.children[0].getZ3()} + {self.children[1].getZ3()}'
                elif self.val in ['AVG', 'COUNT', 'MAX', 'MIN', 'SUM']:
                    self.z3_formula = f'{self.val}({self.children[0].getZ3()})'
                else:
                    raise NotImplementedError(f'{self.val} is not a supported operator.')

        elif self.type == 'var':
            self.z3_formula = self.val

        elif self.type == 'const':
            self.z3_formula = self.val if self.val.isnumeric() else f'StringVal({self.val})'

        else:
             raise NotImplementedError(f'Type {self.type} is not supported.')

        return self.z3_formula
        

    def get_size(self):
        """Get size of the subtree, note that atomic predicate has size 1.
        ### Input: none
        ### Return:
            int: the size of the subtree, all atomic predicates are of size 1
        """
        if self.size > 0:
            return self.size

        if self.type == 'pred':
            self.size = 1
            return self.size

        self.size = sum([c.get_size() for c in self.children]) + 1
        return self.size


    def __repr__(self):
        return self.getZ3()
    def __str__(self):
        return self.getZ3()

