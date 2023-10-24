# built-in packages
import json
from copy import deepcopy

# extra packages
from z3 import *
import networkx as nx

# project packages
# from global_var_tpc import *
from global_var_beers import *




class QueryInfo(object):
    """
    A class used to flatten a query and collect information to track the flattened query

    Attributes (* indicates not needed for now, but might be useful in the future)
    ----------
    schema: dict
        table name --> [[type], [attr]]

    data: dict
        json object of the entire analyze data from Calcite
    xtree: dict
        xtree of the query (body if root node of xtree is WITH, else just same as xtree)
    with_items: dict
        XWithItemNode_id --> original WITH table xtree 
    all_with_reference: dict
        (select_block_xid, table alias in block) --> WITH table xtree (with renamed id)
    with_table_id: int
        with block unique id, also can be used to count total times all WITH tables get referenced

    std_alias_id: int
        standardized table id, each table is renamed to a std alias in the form of "t_id"
    std_alias_id
        a SQL analyzer backed by Calcite
    table_to_alias: dict
        table name in schema --> [std alias of tables that appear in query and belong to this schema]
    std_alias_to_info: dict
        std table alias --> (its XSelectNode id, schema table name, original alias assigned by user)
    *info_to_std_alias: dict
        (XSelectNode id, original alias assigned by user) --> std table alias

    attr_trace: dict
        (XSelectNode id, original table alias, column name) --> (XSelectNode id, original table alias, column name)
        key is the attr in outer query, value is the same attr in inner query. Need to keep this mapping when an attr
        is lifted to the outer query via FROM clause
    flatten_where_trees: xnode
        Binary syntax tree that combine all WHERE clauses in the query
    flatten_groupby_exprs: list
        deepcopy of the list of groupby expression tree
    flatten_having: xtree
        deepcopy of the xtree that contains the outermost having syntax tree
    flatten_select: list
        deepcopy of the xtree that contains the list expression in SELECT clause


    Methods
    -------
    get_std_table_alias()
    parse_table_attr(xnode):
    parse_from_table_helper(select_xid, xnode):
    add_new_table_std_alias(select_xid, table, block_alias):
    add_table_attr_trace(xnode, outer_select_xid, block_alias):
    parse_where_table_helper(xnode)
    parse_all_where_trees(xnode)
    parse_where_in_from(xnode, select_xid)
    parse_where_in_where(xnode)
    parse_expr_no_subquery(xnode, select_xid) 
    rename_new_with_table(xnode):
    """
    def __init__(self, query, analyzer=default_analyzer, schema=db_schema):
        self.schema = schema
        self.analyzer = analyzer
        self.data = json.loads(str(self.analyzer.analyzeToJson(query)))
        self.xtree = None
        self.with_items = {}  # id --> with obj
        self.all_with_reference = {}  # outer_select_xid, alias --> with obj
        self.std_alias_id = 0
        self.with_table_id = 0
        self.table_to_alias = {}
        self.std_alias_to_info = {}
        self.attr_trace = {}
        # self.info_to_std_alias = {}
        # print(json.dumps(self.data))

        # identify whether root node of xtree is WITH, SELECT or SETOP or ERROR
        self.prelim_scan_analyze_data()

        # dive into FROM tables
        self.parse_table_attr(self.xtree)

        self.flatten_where_trees = self.parse_all_where_trees(self.xtree)
        self.flatten_groupby_exprs = deepcopy(self.xtree['group_by_exprs']) if self.xtree['group_by_exprs'] else []
        self.flatten_having = deepcopy(self.xtree['having_cond']) if self.xtree['having_cond'] else None
        self.flatten_select = deepcopy(self.xtree['select_exprs']) if self.xtree['select_exprs'] else []

        for expr in self.flatten_groupby_exprs:
            self.parse_expr_no_subquery(expr)
        for i in range(len(self.flatten_select)):
            # first check if the ref column is an expr from a subquery
            if self.flatten_select[i]['type'] == 'XColumnRefNode':
                table, col = self.flatten_select[i]['sql_string'].split('.')
                cur = self.attr_trace[(self.flatten_select[i]['XSelectNode_id'], table, col)]
                if cur[-1] == 'expr':
                    self.flatten_select[i] = deepcopy(cur[1])
                    self.parse_expr_no_subquery(self.flatten_select[i])
            else:
                self.parse_expr_no_subquery(self.flatten_select[i])
        self.parse_expr_no_subquery(self.flatten_having)


    def prelim_scan_analyze_data(self):
        """
        Quick look into data and make sure it is valid.
        Record WITH clauses and set xtree to the first SELECT context.
        """
        if 'error' in self.data:
            raise RuntimeError(f'QueryInfo: prelim_scan_analyze_data: Query syntax error: {self.data["message"]}.')
        elif self.data['xtree']['type'] == 'XWithNode':
            self.xtree = self.data['body']
            for item in self.data['with_items']:
                self.with_items[item['id']] = self.data['xtree']['with_items']
        elif self.data['xtree']['type'] == 'XSelectNode':
            self.xtree = self.data['xtree']
        else:
            raise NotImplementedError(f'QueryInfo: prelim_scan_analyze_data: node {self.data["type"]} not supported.')


    def get_std_table_alias(self):
        """
        Return a unique standardized table id.
        Get the next id for std table alias.
        """
        id = self.std_alias_id
        self.std_alias_id += 1
        return 't_' + str(id)

    
    def parse_table_attr(self, xnode):
        """
        Sort out aliases for ALL tables and their attributes in the entire query.
        Traverse the xtree to rename all tables to std alias, and collect attribute tracing info.
        After running, std_alias_id, table_to_alias, std_alias_to_info, *info_to_std_alias and attr_trace
        should be populated.
        """
        self.parse_from_table_helper(xnode['id'], xnode['from_expr'])
        self.parse_where_table_helper(xnode['where_cond'])


    def parse_from_table_helper(self, select_xid, xnode):
        """
        Parse the tables in a FROM clause (note that a table might be a derived table).
        Subroutine of parse_table_attr, collect information in subqueries in FROM.
        """
        # preorder traversal
        stack = []
        cur = xnode
        while len(stack) or cur != None:
            while cur != None:

                if cur['type'] == 'XJoinNode':
                    if cur['join_type'] != 'COMMA':
                        raise NotImplemented(f'QueryInfo: currently do not handle JOIN type: {cur["join_type"]}. If using INNER JOIN, please do cross join and place the join condition in WHERE.')
                    stack.append(cur['right'])
                    cur = cur['left']

                # single in-db table without renaming
                elif cur['type'] == 'XTableRefNode' and cur['in_database']:
                    self.add_new_table_std_alias(select_xid, cur['name'], cur['name'])
                    cur = None

                # WITH table
                elif cur['type'] == 'XTableRefNode' and not cur['in_database']:
                    with_table = deepcopy(self.with_items[cur['XWithItemNode_id']])
                    self.rename_new_with_table(with_table)
                    self.with_table_id += 1
                    self.all_with_reference[(select_xid, cur['name'])] = with_table
                    self.parse_table_attr(with_table)
                    self.add_table_attr_trace(self.with_items[cur['operand']['XWithItemNode_id']], select_xid, cur['name'])

                # renamed table
                elif cur['type'] == 'XTableRenameNode':
                    new_name = cur['new_name']
                    # 3 scenarios
                    # derived table, XSelectNode
                    if cur['operand']['type'] == 'XSelectNode':
                        self.parse_table_attr(cur['operand'])
                        self.add_table_attr_trace(cur['operand'], select_xid, new_name)

                    # WITH table
                    elif cur['operand']['type'] == 'XTableRefNode' and not cur['operand']['in_database']:
                        with_table = deepcopy(self.with_items[cur['operand']['XWithItemNode_id']])
                        self.rename_new_with_table(with_table)
                        self.with_table_id += 1
                        self.all_with_reference[(select_xid, new_name)] = with_table
                        self.parse_table_attr(with_table)
                        self.add_table_attr_trace(with_table, select_xid, new_name)

                    # in-db table, XTableRefNode
                    else:
                        self.add_new_table_std_alias(select_xid, cur['operand']['name'], new_name)
                    
                    cur = None
 
            if len(stack) > 0:
                cur = stack[-1]
                stack.pop()


    def parse_where_table_helper(self, xnode):
        """
        Parse tables in the correlated subqueries.
        Subroutine of parse_table_attr, collect information in subqueries in WHERE.
        """
        if not xnode:
            return

        # if the where clause contains only an EXISTS subquery
        if xnode['type'] == 'XBasicCallNode' and 'operator_name' in xnode and xnode['operator_name'] == 'EXISTS':
            self.parse_table_attr(xnode['operands'][0])
        elif xnode['type'] == 'XBasicCallNode':
            if 'operands' in xnode:
                for i in range(len(xnode['operands'])):
                    if 'operator_name' in xnode['operands'][i] and xnode['operands'][i]['operator_name'] == 'EXISTS':
                        self.parse_table_attr(xnode['operands'][i]['operands'][0])
                    elif 'operator_name' in xnode['operands'][i] and xnode['operands'][i]['operator_name'] == 'IN':
                        raise NotImplementedError('test_info: parse_where_table_helper: Please use EXISTS instead of IN.')
                    else:
                        self.parse_where_table_helper(xnode['operands'][i])


    def parse_all_where_trees(self, xnode):
        """
        Parse ALL WHERE predicates in ALL possible places (FROM, WHERE), connect them into one WHERE tree.
        Traverse the xtree to return the hierarchical list for flatten where tree.
        """
        from_trees = self.parse_where_in_from(xnode['from_expr'], xnode['id'])
        
        where_copy = deepcopy(xnode['where_cond']) if xnode['where_cond'] else None

        # ignore the EXISTS operator if where clause only contains one EXISTS subquery
        while where_copy and where_copy['type'] == 'XBasicCallNode' and 'operator_name' in where_copy and where_copy['operator_name'] == 'EXISTS':
            where_copy = where_copy['operands'][0]['where_cond']
        if where_copy:
            where_copy['xid'] = where_copy['id']     # insert the xnode id to identify the context for building syntax tree later
        # print(json.dumps(where_copy))
        self.parse_where_in_where(where_copy)
        # print(json.dumps(where_copy))

        if from_trees and where_copy:
            new_root = {
                'type': 'XConnector',
                'operator_name': 'AND',
                'operands': [where_copy]
            }
        
            for t in from_trees:
                if len(new_root['operands']) == 2:
                    tp = new_root
                    new_root = {
                        'type': 'XConnector',
                        'operator_name': 'AND',
                        'operands': [tp]
                    }
                else:
                    new_root['operands'].append(t)

            return new_root

        elif from_trees:
            if len(from_trees) == 1:
                return from_trees[0]
            
            new_root = {
                'type': 'XConnector',
                'operator_name': 'AND',
                'operands': []
            }

            for t in from_trees:
                if len(new_root['operands']) == 2:
                    tp = new_root
                    new_root = {
                        'type': 'XConnector',
                        'operator_name': 'AND',
                        'operands': [tp]
                    }
                else:
                    new_root['operands'].append(t)
            return new_root

        elif where_copy:
            return where_copy
        
        return None
    

    def parse_where_in_from(self, xnode, block_select_xid):
        """
        Extract all WHERE predicates from the WHERE clause of a dervied table (WITH or subquery).
        Subroutine of parse_all_where_trees, return a list of xtree for each subqueries WHERE in current FROM.
        """
        res = []
        if xnode['type'] == 'XJoinNode':
            if xnode['join_type'] != 'COMMA':
                raise NotImplemented(f'test_info: currently do not handle JOIN type: {xnode["join_type"]}')
            res += self.parse_where_in_from(xnode['left'], block_select_xid)
            res += self.parse_where_in_from(xnode['right'], block_select_xid)
        elif xnode['type'] == 'XTableRenameNode' and xnode['operand']['type'] == 'XSelectNode':
            tp_node = self.parse_all_where_trees(xnode['operand'])
            if tp_node:
                res.append(tp_node)

        # two situations for WITH table (renamed or not renamed)
        elif xnode['type'] == 'XTableRenameNode' and xnode['operand']['type'] == 'XTableRefNode' and not xnode['operand']['in_database']:
            tp_node = self.parse_all_where_trees(self.all_with_reference[(block_select_xid, xnode['new_name'])])
            if tp_node:
                res.append(tp_node)

        elif 'operand' in xnode and xnode['operand']['type'] == 'XTableRefNode' and not xnode['operand']['in_database']:
            tp_node = self.parse_all_where_trees(self.all_with_reference[(block_select_xid, xnode['name'])])
            if tp_node:
                res.append(tp_node)

        return res


    def parse_where_in_where(self, xnode):
        """
        Extract all WHERE predicates from WHERE clause of correlated subqueries.
        Subroutine of parse_all_where_trees, return a xtree for the current WHERE clause.
        """
        if not xnode:
            return

        # where clause contains other predicates or has at least one logical operator
        if xnode['type'] == 'XBasicCallNode':
            if 'operands' in xnode:
                for i in range(len(xnode['operands'])):

                    # check if attr is a projected expr from inner sub-query, adjust the xtree for convenience for future parsing
                    if xnode['operands'][i]['type'] == 'XColumnRefNode':
                        table, col = xnode['operands'][i]['sql_string'].split('.')
                        cur = self.attr_trace[(xnode['operands'][i]['XSelectNode_id'], table, col)]
                        if cur[-1] == 'expr':
                            xnode['operands'][i] = deepcopy(cur[1])
                            self.parse_where_in_where(xnode['operands'][i])

                    # correlated subquery, replace the current subtree with the WHERE xtree of inner query for future parsing
                    elif 'operator_name' in xnode['operands'][i] and xnode['operands'][i]['operator_name'] == 'EXISTS':
                        xnode['operands'][i] = self.parse_all_where_trees(xnode['operands'][i]['operands'][0])

                    # TODO: support IN later
                    elif 'operator_name' in xnode['operands'][i] and xnode['operands'][i]['operator_name'] == 'IN':
                        raise NotImplementedError('test_info: parsing_where_in_where: Please use EXISTS instead of IN.')

                    # any other case, keep recursing down
                    else:
                        self.parse_where_in_where(xnode['operands'][i])

            else:
                # don't handle this operator
                raise NotImplementedError(f'test_info: parsing_where_in_where: Operator < {xnode["operator_name"]} > is not currently supported.')


    def parse_expr_no_subquery(self, xnode):
        """
        Used to parse the expr syntax tree for the SELECT, GROUP BY and HAVING of the outermost query block
        Similar to parse_where_in_where but no need to consider subqueries
        """
        if not xnode:
            return

        if xnode['type'] == 'XBasicCallNode':
            if 'operands' in xnode:
                for i in range(len(xnode['operands'])):
                    # check if attr is a projected expr from inner sub-query, adjust the xtree for convenience for future parsing
                    if xnode['operands'][i]['type'] == 'XColumnRefNode':
                        table, col = xnode['operands'][i]['sql_string'].split('.')
                        cur = self.attr_trace[(xnode['operands'][i]['XSelectNode_id'], table, col)]
                        if cur[-1] == 'expr':
                            xnode['operands'][i] = deepcopy(cur[1])
                            self.parse_expr_no_subquery(xnode['operands'][i], cur[0])
                    
                    # operator, leave it untouched
                    else:
                        self.parse_expr_no_subquery(xnode['operands'][i])
        elif xnode['type'] == 'XColumnRenameNode':
            self.parse_expr_no_subquery(xnode['operand'])


    def add_new_table_std_alias(self, select_xid, table, block_alias):
        """
        Map the table standard alias to (select_xid, original in-db table, table alias in the select_xid context).
        Also keep track of attributes in the table.
        Create new std alias for a table, add all its attributes to attr_trace.
        """
        std_alias = self.get_std_table_alias()

        # for the original in-db table, register its new std_alias, it could have multiple across different context
        if table in self.table_to_alias.keys():
            self.table_to_alias[table].append(std_alias)
        else:
            self.table_to_alias[table] = [std_alias]
        self.std_alias_to_info[std_alias] = (select_xid, table, block_alias)
        # self.info_to_std_alias[(select_xid, new_name)] = std_alias
        # given xid, table alias in the context and its attribute, we should be able to find its std_alias
        for attr in self.schema[table][1]:
            self.attr_trace[(select_xid, block_alias, attr)] = (std_alias, attr, 'std')
    

    def add_table_attr_trace(self, xnode, outer_select_xid, block_alias):
        """
        For any SELECT expressions in any SELECT context, map its alias in the outer context to its alias in inner context.
        Connect the inner query projection with outer query attributes, store such info in attr_trace.
        """
        for expr in xnode['select_exprs']:
            inblock_table_alias, col = '', ''
            if expr['type'] == 'XColumnRenameNode':
                if expr['operand']['type'] == 'XBasicCallNode':
                    self.attr_trace[(outer_select_xid, block_alias, expr['new_name'])] = (xnode['id'], expr['operand'], 'expr')
                    continue
                                
                inblock_table_alias, col = expr['operand']['sql_string'].split('.')
                                
            elif expr['type'] == 'XColumnRefNode':
                inblock_table_alias, col = expr['sql_string'].split('.')
            self.attr_trace[(outer_select_xid, block_alias, col)] = (xnode['id'], inblock_table_alias, col, 'intermediate')


    def rename_new_with_table(self, xnode):
        """
        Make a reference of WITH table unique by adding a unique number to the end.
        Change all xtree node id (rooted at xnode) to the form "id_with_table_id" to distinguish among WITH references.
        """
        if not xnode:
            return

        xnode['id'] += '_' + str(self.with_table_id)
        if xnode['type'] == 'XSelectNode':
            self.rename_new_with_table(xnode['from_expr'])
            for expr in xnode['select_exprs']:
                self.rename_new_with_table(expr)
            self.rename_new_with_table(xnode['where_cond'])
            if xnode['group_by_exprs'] or xnode['having_cond']:
                raise NotImplementedError(f'QueryInfo: rename_new_with_table: do not support GROUP BY and HAVING in WITH tables.')
        elif xnode['type'] == 'XJoinNode':
            self.rename_new_with_table(xnode['left'])
            self.rename_new_with_table(xnode['right'])
        elif xnode['type'] == 'XBasicCallNode':
            for operand in xnode['operands']:
                self.rename_new_with_table(operand)
        elif xnode['type'] == 'XTableRenameNode' or xnode['type'] == 'XColumnRenameNode':
            self.rename_new_with_table(xnode['operand'])
        elif xnode['type'] in ['XTableRefNode', 'XColumnRefNode', 'XLiteralNode']:
            xnode['id'] += '_' + str(self.with_table_id)
        else:
            raise NotImplementedError(f'QueryInfo: rename_new_with_table: do not support {xnode["type"]} in WITH clause.')
        
    




class MappingInfo(object):
    """
    A class used to generate and store query mapping

    Attributes
    ----------
    schema: dict
        table name --> [[type], [attr]]
    table_mappings: list
        a list of pairs of dict, one for each query, table alias --> mutual alias
    table_mappings_reverse: list
        a list of pairs of dict, one for each query, mutual alias --> table alias
    z3_var_lookup: dict
        table alias --> [z3 var instance, one for each attr in table]
    q1_info: QueryInfo
    q2_info: QueryInfo


    Methods
    -------
    reset(q1, q2)
        use the same database but analyze two new queries
    declare_z3_var()
        given a set of table names, populate z3_var_lookup
    find_all_mappings()
        find all possible table mappings between q1 and q2, store in table_mappings
    find_all_mappings_reverse()
        find the reverse mapping (mutual alias --> query alias)
    permute_mapping(t1, t2): list of lists of tuples
        given two lists of tables, find all possible mappings between them

    std_alias_to_info
    """
    def __init__(self, q1_info: QueryInfo, q2_info: QueryInfo, schema=db_schema):
        """
        Parameters
        ----------
        q1_info: QueryInfo
            q1 info object
        q2_info: QueryInfo
            q2 info object
        schema: dict
            table --> [[type], [attr]]
        """
        self.schema = schema
        self.table_mapping = [{}, {}]
        self.table_mapping_reverse = [{}, {}]
        self.z3_var_lookup = {}
        self.q1_info = q1_info
        self.q2_info = q2_info

        self.original_to_std_alias = [{}, {}]
        self.std_alias_to_table = [{}, {}]
        self.alias_to_table = [{}, {}]
        self.attr_equiv_class = [{}, {}]

        self.where_sigs = [{}, {}]      # table alias -> (attr, op) -> { equiv class attrs }
        self.groupby_sigs = [{}, {}]    # table alias -> { attr in group by}
        self.select_sigs = [{}, {}]     # table alias -> attr -> { index in SELECT exprs}

        self.mapping_check()
        self.pre_mapping()
        self.init_sigs()
        self.scan_attr_equiv_class(self.q1_info.xtree['where_cond'], self.attr_equiv_class[0])
        self.scan_attr_equiv_class(self.q2_info.xtree['where_cond'], self.attr_equiv_class[1])
        self.populate_where_signature(self.q1_info.xtree['where_cond'], 0)
        self.populate_where_signature(self.q2_info.xtree['where_cond'], 1)
        self.populate_groupby_signature(self.q1_info.xtree['group_by_exprs'], 0)
        self.populate_groupby_signature(self.q2_info.xtree['group_by_exprs'], 1)
        self.populate_select_signature(self.q1_info.xtree['select_exprs'], 0)
        self.populate_select_signature(self.q2_info.xtree['select_exprs'], 1)
        self.determine_mapping()
        # self.find_all_mappings()
        # self.find_all_mappings_reverse()
        self.declare_z3_var()


    def mapping_check(self):
        if len(self.q1_info.table_to_alias) != len(self.q2_info.table_to_alias):
            diff1 = set(self.q1_info.table_to_alias.keys()) - set(self.q2_info.table_to_alias.keys())
            diff2 = set(self.q2_info.table_to_alias.keys()) - set(self.q1_info.table_to_alias.keys())
            if diff1 and diff2:
                raise ValueError(f'Table {diff1} are missing and Table {diff2} are redundant.')
            elif diff1:
                raise ValueError(f'Table {diff1} are missing.')
            elif diff2:
                raise ValueError(f'Table {diff2} are redundant.')

        # check same multiset of table
        for k, v in self.q1_info.table_to_alias.items():
            w = self.q2_info.table_to_alias[k]
            if len(v) != len(w):
                raise ValueError(f'The table {k} should be referenced {len(v)} times, but {len(self.q2_info.table_to_alias[k])} times in user query!')
            

    def pre_mapping(self):

        for k, v in self.q1_info.std_alias_to_info.items():
            self.original_to_std_alias[0][v[2]] = k
            self.std_alias_to_table[0][k] = v[1]
        for k, v in self.q2_info.std_alias_to_info.items():
            self.original_to_std_alias[1][v[2]] = k
            self.std_alias_to_table[1][k] = v[1]

        for k, v in self.q1_info.table_to_alias.items():
            w = self.q2_info.table_to_alias[k]
            if len(v) == len(w) and len(v) == 1:
                mutual_name = f'{k}_0'
                self.table_mapping[0][v[0]] = mutual_name
                self.table_mapping[1][w[0]] = mutual_name
                self.table_mapping_reverse[0][mutual_name] = v[0]
                self.table_mapping_reverse[1][mutual_name] = w[0]
            else:
                for alias in v:
                    self.alias_to_table[0][alias] = k
                for alias in w:
                    self.alias_to_table[1][alias] = k

        
    def init_sigs(self):
        for alias, table in self.alias_to_table[0].items():
            self.groupby_sigs[0][alias] = set()
            self.where_sigs[0][alias] = {}
            self.select_sigs[0][alias] = {}
            for attr in self.schema[table][1]:
                self.select_sigs[0][alias][attr] = set()
                for op in ['=', '<>', '<', '<=', '>', '>=']:
                    self.where_sigs[0][alias][(attr, op)] = set()

        for alias, table in self.alias_to_table[1].items():
            self.groupby_sigs[1][alias] = set()
            self.where_sigs[1][alias] = {}
            self.select_sigs[1][alias] = {}
            for attr in self.schema[table][1]:
                self.select_sigs[1][alias][attr] = set()
                for op in ['=', '<>', '<', '<=', '>', '>=']:
                    self.where_sigs[1][alias][(attr, op)] = set()


    def scan_attr_equiv_class(self, where_tree: dict, attr_equiv_class: dict):
        if where_tree['type'] == 'XBasicCallNode' and where_tree['operator_name'] == '=':
            if where_tree['operands'][0]['type'] != 'XBasicCallNode' and where_tree['operands'][1]['type'] != 'XBasicCallNode':
                left, right = where_tree['operands'][0]['sql_string'], where_tree['operands'][1]['sql_string']
                if left in attr_equiv_class:
                    attr_equiv_class[left].append(right)
                else:
                    attr_equiv_class[left] = [right]
                if right in attr_equiv_class:
                    attr_equiv_class[right].append(left)
                else:
                    attr_equiv_class[right] = [left]
            else:
                return
        elif 'operands' in where_tree:
            for operand in where_tree['operands']:
                self.scan_attr_equiv_class(operand, attr_equiv_class)

    
    def extract_attrs(self, expr: dict):
        if expr['type'] in ['XColumnRefNode', 'XLiteralNode']:
            return [expr['sql_string']]
        
        res = []
        for operand in expr['operands']:
            res += self.extract_attrs(operand)
        return res


    def where_add_attr_to_set_helper(self, target: set, visited: set, cur: str, qid: int):
        tp = cur.split('.')
        visited.add(cur)
        if len(tp) > 1:
            target.add(self.std_alias_to_table[qid][self.original_to_std_alias[qid][tp[0]]].lower())
        else:
            target.add(cur)
        if cur in self.attr_equiv_class[qid]:
            for attr in self.attr_equiv_class[qid][cur]:
                if attr not in visited:
                    self.where_add_attr_to_set_helper(target, visited, attr, qid)
        visited.remove(cur)


    def populate_where_signature(self, where_tree: dict, qid: int):
        if where_tree['type'] == 'XBasicCallNode' and where_tree['operator_name'] in ['=', '<>', '<', '<=', '>', '>=']:
            left = self.extract_attrs(where_tree['operands'][0])
            right = self.extract_attrs(where_tree['operands'][1])

            op = where_tree['operator_name']
            if op in ['=', '<>']:
                for i in range(len(left)):
                    for j in range(len(right)):
                        visited = set()
                        attr1 = left[i].split('.')
                        attr2 = right[j].split('.')
                        if len(attr1) > 1:
                            attr1_std = self.original_to_std_alias[qid][attr1[0]]
                            if attr1_std in self.where_sigs[qid]:
                                self.where_add_attr_to_set_helper(self.where_sigs[qid][attr1_std][(attr1[1], op)], visited, right[j], qid)
                        if len(attr2) > 1:
                            attr2_std = self.original_to_std_alias[qid][attr2[0]]
                            if attr2_std in self.where_sigs[qid]:
                                self.where_add_attr_to_set_helper(self.where_sigs[qid][attr2_std][(attr2[1], op)], visited, left[i], qid)
            else:
                reverse_op = None
                if op in ['<', '<=']:
                    reverse_op = '>' if op == '<' else '>='
                elif op in ['>', '>=']:
                    reverse_op = '<' if op == '>' else '<='
                else:
                    return
                
                for i in range(len(left)):
                    for j in range(len(right)):
                        visited = set()
                        attr1 = left[i].split('.')
                        attr2 = right[j].split('.')
                        if len(attr1) > 1:
                            attr1_std = self.original_to_std_alias[qid][attr1[0]]
                            if attr1_std in self.where_sigs[qid]:
                                self.where_add_attr_to_set_helper(self.where_sigs[qid][attr1_std][(attr1[1], op)], visited, right[j], qid)
                        if len(attr2) > 1:
                            attr2_std = self.original_to_std_alias[qid][attr2[0]]
                            if attr2_std in self.where_sigs[qid]:
                                self.where_add_attr_to_set_helper(self.where_sigs[qid][attr2_std][(attr2[1], reverse_op)], visited, left[i], qid)
            
        if where_tree['type'] == 'XBasicCallNode' and where_tree['operator_name'] in ['AND', 'OR', 'NOT']:
            for operand in where_tree['operands']:
                self.populate_where_signature(operand, qid)


    def groupby_add_to_other_table_helper(self, visited: set, cur: str, qid: int):
        visited.add(cur)
        if cur in self.attr_equiv_class[qid]:
            for attr in self.attr_equiv_class[qid][cur]:
                if attr not in visited:
                    tp = attr.split('.')
                    if len(tp) > 1:
                        tp_std_alias = self.original_to_std_alias[qid][tp[0]]
                        if tp_std_alias in self.groupby_sigs[qid]:
                            self.groupby_sigs[qid][tp_std_alias].add(tp[1])
                    self.groupby_add_to_other_table_helper(visited, attr, qid)

    def populate_groupby_signature(self, exprs: list, qid: int):
        for expr in exprs:
            attrs = self.extract_attrs(expr)
            for attr in attrs:
                tp = attr.split('.')
                if len(tp) == 2:
                    tp_std_alias = self.original_to_std_alias[qid][tp[0]]
                    if tp_std_alias in self.groupby_sigs[qid]:
                        self.groupby_sigs[qid][tp_std_alias].add(tp[1])
                    visited = set()
                    self.groupby_add_to_other_table_helper(visited, attr, qid)


    def select_add_to_other_table_helper(self, visited: set, cur: str, idx: int, qid: int):
        visited.add(cur)
        if cur in self.attr_equiv_class[qid]:
            for attr in self.attr_equiv_class[qid][cur]:
                if attr not in visited:
                    tp = attr.split('.')
                    if len(tp) > 1:
                        tp_std_alias = self.original_to_std_alias[qid][tp[0]]
                        if tp_std_alias in self.select_sigs[qid]:
                            self.select_sigs[qid][tp_std_alias][tp[1]].add(idx)
                    self.select_add_to_other_table_helper(visited, attr, idx, qid)
    
    def populate_select_signature(self, exprs: list, qid: int):
        for i, expr in enumerate(exprs):
            attrs = self.extract_attrs(expr)
            for attr in attrs:
                tp = attr.split('.')
                if len(tp) == 2:
                    if self.original_to_std_alias[qid][tp[0]] in self.select_sigs[qid]:
                        self.select_sigs[qid][self.original_to_std_alias[qid][tp[0]]][tp[1]].add(i)
                    visited = set()
                    self.select_add_to_other_table_helper(visited, attr, i, qid)


    def compute_sig_diff(self, t1: str, t2: str):
        where_jaccard, group_jaccard, select_jaccard = 0, 0, 0
        for k, v in self.where_sigs[0][t1].items():
            if len(v) == 0 and len(self.where_sigs[1][t2][k]) == 0:
                where_jaccard += 1
            elif len(v) > 0 or len(self.where_sigs[1][t2][k]) > 0:
                where_jaccard += len(v.intersection(self.where_sigs[1][t2][k])) / len(v.union(self.where_sigs[1][t2][k]))
        where_jaccard /= len(self.where_sigs[0][t1])
        group_jaccard += len(self.groupby_sigs[0][t1].intersection(self.groupby_sigs[1][t2])) / len(self.groupby_sigs[0][t1].union(self.groupby_sigs[1][t2]))
        for k, v in self.select_sigs[0][t1].items():
            if len(v) == 0 and len(self.select_sigs[1][t2]) == 0:
                select_jaccard += 1
            elif len(v) > 0 or len(self.select_sigs[1][t2]) > 0:
                select_jaccard += len(v.intersection(self.select_sigs[1][t2])) / len(v.union(self.select_sigs[1][t2]))
        select_jaccard /= len(self.select_sigs[0][t1])
        return round(where_jaccard + group_jaccard + select_jaccard, 2)
    
    def determine_mapping(self):
        for k, v in self.q1_info.table_to_alias.items():
            w = self.q2_info.table_to_alias[k]
            if len(v) == len(w) and len(v) == 1:
                continue
            
            offset = len(v)
            G = nx.Graph()
            edges = []
            for i, t1 in enumerate(v):
                for j, t2 in enumerate(w):
                    edges.append((i, j + offset, self.compute_sig_diff(t1, t2)))
            G.add_weighted_edges_from(edges)
            res = nx.max_weight_matching(G)
            for i, e in enumerate(res):
                e = list(e)
                e.sort()
                mutual_name = f'{k}_{i}'
                self.table_mapping[0][v[e[0]]] = mutual_name
                self.table_mapping[1][w[e[1] - offset]] = mutual_name
                self.table_mapping_reverse[0][mutual_name] = v[e[0]]
                self.table_mapping_reverse[1][mutual_name] = w[e[1] - offset]

                

    def declare_z3_var(self):
        # for each table in multiset, 
        for k, v in self.table_mapping[0].items():
            # find the table name in schema, name in format "table_id"
            table = v.split('_')[0]
            # declare z3 var for each attr
            for i in range(len(self.schema[table][0])):
                if i == 0:
                    self.z3_var_lookup[v] = []
                if self.schema[table][0][i] == 'string':
                    self.z3_var_lookup[v].append(String(f'{v}.{self.schema[table][1][i]}'))
                elif self.schema[table][0][i] == 'int':
                    self.z3_var_lookup[v].append(Int(f'{v}.{self.schema[table][1][i]}'))
                elif self.schema[table][0][i] == 'float':
                    self.z3_var_lookup[v].append(Real(f'{v}.{self.schema[table][1][i]}'))
            
    
    def permute_mappings(self, t1, t2):
        if len(t1) == 1:
            return [[(t1[0], t2[0])]]

        res = []
        has_been_equal = False  # for i==j case, we can only produce such mapping once
        for i in range(len(t1)):
            for j in range(len(t2)):
                # assume t1[i] -> t2[j]

                # guard against duplicate mapping
                if i >= j and has_been_equal:
                    continue

                if i == j:
                    has_been_equal = True

                # map the rest of tables
                t1_remain, t2_remain = t1[:i] + t1[i+1:], t2[:j] + t2[j+1:]
                sub_mapping = self.permute_mappings(t1_remain, t2_remain)
                for k in range(len(sub_mapping)):
                    sub_mapping[k].append((t1[i], t2[j]))   # add the t1[i] -> t2[j]
                res += sub_mapping
        return res


