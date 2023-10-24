# extra packages
import jpype
import jpype.imports
jpype.startJVM(classpath=['sqlanalyzer.jar'])
from edu.duke.cs.irex.sqlanalyzer import Analyzer


# pre-define a schema here
db_schema = {
    'part': [['int', 'string', 'string', 'string', 'string', 'int', 'string', 'float', 'string'],
                ['p_partkey', 'p_name', 'p_mfgr', 'p_brand', 'p_type', 'p_size', 'p_container', 'p_retailprice', 'p_comment']
    ],
    'supplier': [['int', 'string', 'string', 'int', 'string', 'float', 'string'],
                    ['s_suppkey', 's_name', 's_address', 's_nationkey', 's_phone', 's_acctbal', 's_comment']
    ],
    'partsupp': [['int', 'int', 'int', 'float', 'string'],
                    ['ps_partkey', 'ps_suppkey', 'ps_availqty', 'ps_supplycost', 'ps_comment']
    ],
    'customer': [['int', 'string', 'string', 'int', 'string', 'float', 'string', 'string'],
                    ['c_custkey', 'c_name', 'c_address', 'c_nationkey', 'c_phone', 'c_acctbal', 'c_mktsegment', 'c_comment']
    ],
    'nation': [['int', 'string', 'int', 'string'], 
                ['n_nationkey', 'n_name', 'n_regionkey', 'n_comment']
    ],
    'region': [['int', 'string', 'string'],
                ['r_regionkey', 'r_name', 'r_comment']
    ],
    'lineitem': [['int', 'int', 'int', 'int', 'float', 'float', 'float', 'float', 'string', 'string', 'int', 'int', 'int', 'string', 'string', 'string'],
                    ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct', 'l_shipmode', 'l_comment']
    ],
    'orders': [['int', 'int', 'string', 'float', 'int', 'string', 'string', 'int', 'string'],
                ['o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate', 'o_orderpriority', 'o_clerk', 'o_shippriority', 'o_comment']
    ]
}


# pre-define an analyzer
# default_analyzer = Analyzer('localhost', 'beers', 'postgres', 'change_me')
# default_analyzer = Analyzer('localhost', 'dblp', 'postgres', 'postgres')
default_analyzer = Analyzer('localhost', 'tpc', 'postgres', 'postgres')


# predefine supported operators
log_ops = ['AND', 'OR', 'NOT']
cmp_ops = ['>', '>=', '<', '<=', '=', '!=', '<>']
agg_ops = ['COUNT', 'MAX', 'MIN', 'AVG', 'SUM']
ari_ops = ['+', '-', '*', '/', '||']