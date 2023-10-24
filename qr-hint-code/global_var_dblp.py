# extra packages
import jpype
import jpype.imports
jpype.startJVM(classpath=['sqlanalyzer.jar'])
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



# pre-define an analyzer
# default_analyzer = Analyzer('localhost', 'beers', 'postgres', 'change_me')
default_analyzer = Analyzer('localhost', 'dblp', 'postgres', 'postgres')


# predefine supported operators
log_ops = ['AND', 'OR', 'NOT']
cmp_ops = ['>', '>=', '<', '<=', '=', '!=', '<>']
agg_ops = ['COUNT', 'MAX', 'MIN', 'AVG', 'SUM']
ari_ops = ['+', '-', '*', '/', '||']