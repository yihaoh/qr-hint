from boolean_parse_tree import *
from itertools import combinations


class RepairSitesIter:
    """
    Iterator that goes to the next set of disjoint subtrees with minimum size

    Attributes
    ----------
    nodes: list
        a list of all nodes in the syntax tree
    all_comb: list of list
        list of all combinations of n nodes
    idx: int
        where we are in all_comb


    Methods
    -------
    next(): list
        return the next set of disjoint subtrees with minimum size
    is_ancestor(a, b): Boolean
        check if a is an ancestor of b
    reset(): 
        reset idx to 0
    get_all_nodes(tree):
        traverse each node in the tree (post-order) and put them in a list
    """

    def __init__(self, subtree: BNode, n: int, p1_size, p2_size):
        self.nodes = []
        self.get_all_nodes(subtree)
        self.all_comb = []
        for i in range(n):
            self.all_comb = self.all_comb + list(combinations(self.nodes, i+1))
        self.all_comb.sort(key=lambda x: sum([y.get_size() for y in x])) #+ (p1_size + p2_size) / (2*len(x)))
        # print('iter len: ', len(self.all_comb))
        self.idx = 0

    
    def next(self):
        while self.idx < len(self.all_comb):
            target = self.all_comb[self.idx]

            # pairwise checking if subtrees are disjoint
            overlap = False
            for i in range(len(target)):
                for j in range(i + 1, len(target)):
                    if self.is_ancestor(target[i], target[j]) or self.is_ancestor(target[j], target[i]):
                        overlap = True
            
            self.idx += 1
            if not overlap:
                return target
                
        return None


    def is_ancestor(self, a, b):
        while b:
            if b == a:
                return True
            b = b.parent
        return False 


    def reset(self):
        self.idx = 0


    def get_all_nodes(self, root):
        if root.type == 'pred':
            self.nodes.append(root)
            return

        for n in root.children:
            self.get_all_nodes(n)

        self.nodes.append(root)