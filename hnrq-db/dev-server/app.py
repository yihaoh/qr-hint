from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
import json
import psycopg2
from psycopg2 import connect, extras

app = Flask(__name__)

import jpype
import jpype.imports

jpype.startJVM(classpath=["sqlanalyzer.jar"])
from edu.duke.cs.irex.sqlanalyzer import Analyzer

dbuser = "hnrq"
dbpassword = "hnrq"
# analyzer = Analyzer('localhost', "beers_db1", dbuser, dbpassword)

db_instances = dict()
db_names = ['beers_db1']
db_metadata = dict()
# pre-fetch database schemas to be passed to frontend
for db in db_names:
    conn_config = {'dbname': db,
                   'user': dbuser,
                   'password': dbpassword,
                   'host': "localhost",
                   'port': "5432"}
    with connect(**conn_config) as conn:
        db_instances[db] = dict()
        db_metadata[db] = dict()
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute("select table_name from information_schema.tables where table_schema = 'public'")
            schema = cur.fetchall()
            for table in schema:
                table_name = table['table_name']
                db_instances[db][table_name] = dict()
                cur.execute(f"SELECT * FROM {table_name}")
                records = cur.fetchall()
                db_instances[db][table_name]['columns'] = cur.column_mapping
                db_instances[db][table_name]['width'] = 0
                for col in db_instances[db][table_name]['columns']:
                    db_instances[db][table_name]['width'] += (len(col) * 15)
                db_instances[db][table_name]['index'] = []
                db_instances[db][table_name]['data'] = []
                for i, record in enumerate(records):
                    db_instances[db][table_name]['index'].append(i)
                    row = []
                    for key in record:
                        row.append(record[key])
                    db_instances[db][table_name]['data'].append(row)
                cur.execute(f"SELECT column_name, data_type \
                                FROM information_schema.columns \
                                WHERE table_name = '{table_name}'")
                meta_records_type = cur.fetchall()
                db_metadata[db][table_name] = dict()
                for meta_record_type in meta_records_type:
                    if meta_record_type['data_type'] == 'character varying':
                        db_metadata[db][table_name][meta_record_type['column_name']] = 'VARCHAR'
                    else:
                        db_metadata[db][table_name][meta_record_type['column_name']] = str(meta_record_type['data_type']).upper()
                cur.execute(f"SELECT kcu.column_name \
                                FROM information_schema.table_constraints tc \
                                JOIN information_schema.key_column_usage kcu \
                                ON tc.constraint_name = kcu.constraint_name \
                                WHERE tc.table_name = '{table_name}' \
                                AND tc.constraint_type = 'PRIMARY KEY'")
                meta_records_pkey = cur.fetchall()
                db_metadata[db][table_name]['pkeys'] = list()
                for meta_record_pkey in meta_records_pkey:
                    db_metadata[db][table_name]['pkeys'].append(meta_record_pkey['column_name'])


@app.route("/db_metadata", methods=['GET', 'POST'])
# @jwt_required()
def get_meta():
    # req = request.environ
    return json.dumps(db_metadata, default=str)


@app.route("/")
def hello():
    return app.send_static_file("index.html")


@app.route("/test", methods=["GET"])
def test():
    conn = psycopg2.connect(database="beers_db1", user="hnrq", password="hnrq", host="localhost", port=5432)
    cur = conn.cursor()
    cur.execute("""SELECT * FROM drinker limit 1""")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    print(rows[0])
    return str(rows[0])


@app.route("/analyze", methods=["POST"])
def analyze():
    user_data = request.get_json()
    # return jsonify({"response": "this is the analyze API."})
    try:
        analyzer = Analyzer("localhost", user_data["db"], dbuser, dbpassword)
        query_context_in_java_json = analyzer.analyzeToJson(user_data["query"], user_data["page_size"])
    except Exception as e:
        return jsonify(error="Unable to process request", message=str(e))
    return str(query_context_in_java_json)


@app.route("/execute_milestone", methods=["POST"])
def execute_milestone():
    if not request.is_json:
        return jsonify(error="request content type not application/json")
    try:
        request_json = request.get_json()
        analyzer = Analyzer("localhost", request_json["db"], dbuser, dbpassword)
    except Exception as e:
        return jsonify(error="Unable to process request", message=str(e))
    # return jsonify({"response": "this is the execute API."})
    code_json = request_json.get("sql", None)
    if code_json is None:
        return jsonify(error="code parameter missing")
    bindings_json = request_json.get("bindings")
    if bindings_json is None:
        return jsonify(error="bindings parameter missing")
    pins_json = request_json.get("pins")
    if pins_json is None:
        return jsonify(error="pins parameter missing")
    context_execd_in_java_json = analyzer.executeMilestoneToJson(json.dumps(code_json), json.dumps(bindings_json), json.dump(pins_json))
    return str(context_execd_in_java_json)


@app.route("/execute_page", methods=["POST"])
def execute_page():
    if not request.is_json:
        return jsonify(error="request content type not application/json")
    try:
        request_json = request.get_json()
        analyzer = Analyzer("localhost", request_json["db"], dbuser, dbpassword)
    except Exception as e:
        return jsonify(error="Unable to process request", message=str(e))
    # return jsonify({"response": "this is the execute API."})
    code_json = request_json.get("sql", None)
    if code_json is None:
        return jsonify(error="code parameter missing")
    bindings_json = request_json.get("bindings")
    if bindings_json is None:
        return jsonify(error="bindings parameter missing")
    filters_json = request_json.get("filters")
    if filters_json is None:
        return jsonify(error="filters parameter missing")
    pins_json = request_json.get("pins")
    if pins_json is None:
        return jsonify(error="pins parameter missing")
    context_execd_in_java_json = analyzer.executePageToJson(json.dumps(code_json), json.dumps(bindings_json), json.dumps(filters_json), json.dumps(pins_json))
    return str(context_execd_in_java_json)


@app.route("/execute_eval", methods=["POST"])
def execute_eval():
    if not request.is_json:
        return jsonify(error="request content type not application/json")
    try:
        request_json = request.get_json()
        analyzer = Analyzer("localhost", request_json["db"], dbuser, dbpassword)
    except Exception as e:
        return jsonify(error="Unable to process request", message=str(e))
    # return jsonify({"response": "this is the execute API."})
    code_json = request_json.get("sql", None)
    if code_json is None:
        return jsonify(error="code parameter missing")
    bindings_json = request_json.get("bindings")
    if bindings_json is None:
        return jsonify(error="bindings parameter missing")
    rows_json = request_json.get("rows")
    if rows_json is None:
        return jsonify(error="rows parameter missing")
    context_execd_in_java_json = analyzer.executeEvalToJson(json.dumps(code_json), json.dumps(bindings_json), json.dumps(rows_json))
    return str(context_execd_in_java_json)


# allow requests from different origin
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response
