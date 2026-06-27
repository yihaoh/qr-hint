# extra packages
import jpype
import jpype.imports
import os

# Get the directory where this file is located
_current_dir = os.path.dirname(os.path.abspath(__file__))
_jar_path = os.path.join(_current_dir, 'sqlanalyzer.jar')

# avoid JVM restart
if not jpype.isJVMStarted():
    jpype.startJVM(classpath=[_jar_path])
from edu.duke.cs.irex.sqlanalyzer import Analyzer


# pre-define a schema here
# db_schema = {
#     'drinker': [['string', 'string'], ['name', 'address']],
#     'bar': [['string', 'string'], ['name', 'address']],
#     'beer': [['string', 'string'], ['name', 'brewer']],
#     'frequents': [['string', 'string', 'int'], ['drinker', 'bar', 'times_a_week']],
#     'serves': [['string', 'string', 'float'], ['bar', 'beer', 'price']],
#     'likes': [['string', 'string'], ['drinker', 'beer']]
# }

db_schema = {
    'inproceedings': [['string', 'string', 'string', 'int', 'string'], ['pubkey', 'title', 'booktitle', 'yearx', 'area']],
    'article': [['string', 'string', 'string', 'int'], ['pubkey', 'title', 'journal', 'yearx']],
    'authorship': [['string', 'string'], ['pubkey', 'author']]
}

# before docker
# default_analyzer = Analyzer('localhost', 'dblp', 'postgres', 'postgres')

_db_host = os.getenv("DB_HOST", "localhost")
default_analyzer = Analyzer(_db_host, 'dblp', 'postgres', 'postgres')


# predefine supported operators
log_ops = ['AND', 'OR', 'NOT']
cmp_ops = ['>', '>=', '<', '<=', '=', '!=', '<>']
agg_ops = ['COUNT', 'MAX', 'MIN', 'AVG', 'SUM']
ari_ops = ['+', '-', '*', '/', '||']