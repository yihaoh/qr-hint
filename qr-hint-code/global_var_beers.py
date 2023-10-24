# extra packages
import jpype
import jpype.imports
jpype.startJVM(classpath=['sqlanalyzer.jar'])
from edu.duke.cs.irex.sqlanalyzer import Analyzer


# pre-define a schema here
db_schema = {
    'Drinker': [['string', 'string'], ['name', 'address']],
    'Bar': [['string', 'string'], ['name', 'address']],
    'Beer': [['string', 'string'], ['name', 'brewer']],
    'Frequents': [['string', 'string', 'int'], ['drinker', 'bar', 'times_a_week']],
    'Serves': [['string', 'string', 'float'], ['bar', 'beer', 'price']],
    'Likes': [['string', 'string'], ['drinker', 'beer']]
}



# pre-define an analyzer
# default_analyzer = Analyzer('localhost', 'beers', 'postgres', 'change_me')
default_analyzer = Analyzer('localhost', 'beers', 'postgres', 'postgres')


# predefine supported operators
log_ops = ['AND', 'OR', 'NOT']
cmp_ops = ['>', '>=', '<', '<=', '=', '!=', '<>']
agg_ops = ['COUNT', 'MAX', 'MIN', 'AVG', 'SUM']
ari_ops = ['+', '-', '*', '/', '||']