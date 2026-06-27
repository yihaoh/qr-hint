# extra packages
import jpype
import jpype.imports
import os

# Get the directory where this file is located
_current_dir = os.path.dirname(os.path.abspath(__file__))
_jar_path = os.path.join(_current_dir, 'sqlanalyzer.jar')

# jpype.startJVM(classpath=["sqlanalyzer.jar"])
# avoid JVM restart
if not jpype.isJVMStarted():
    jpype.startJVM(classpath=[_jar_path])
from edu.duke.cs.irex.sqlanalyzer import Analyzer


# pre-define a schema here
db_schema = {
    "Drinker": [["string", "string"], ["name", "address"]],
    "Bar": [["string", "string"], ["name", "address"]],
    "Beer": [["string", "string"], ["name", "brewer"]],
    "Frequents": [["string", "string", "int"], ["drinker", "bar", "times_a_week"]],
    "Serves": [["string", "string", "float"], ["bar", "beer", "price"]],
    "Likes": [["string", "string"], ["drinker", "beer"]],
}


# pre-define an analyzer
# default_analyzer = Analyzer(host, dbname, username, password)

# origin
# default_analyzer = Analyzer("localhost", "beers", "postgres", "postgres")

# before docker
# default_analyzer = Analyzer("localhost", "beers_db0", "hnrq", "hnrq")

# test on beer
_db_host = os.getenv("DB_HOST", "localhost")
default_analyzer = Analyzer(_db_host, "beers_db0", "hnrq", "hnrq")


# predefine supported operators
log_ops = ["AND", "OR", "NOT"]
cmp_ops = [">", ">=", "<", "<=", "=", "!=", "<>"]
agg_ops = ["COUNT", "MAX", "MIN", "AVG", "SUM", "DISTINCT"]
ari_ops = ["+", "-", "*", "/", "||"]
