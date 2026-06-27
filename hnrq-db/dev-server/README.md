# Dev Server README

This is a simple Flask dev server used to call `sqlanalyzer.jar`. Before starting it up, please make sure the dockerized Postgres is up and running without any errors. Also, please download the [sqlanalyzer](https://users.cs.duke.edu/~yh218/uploads/sqlanalyzer.jar) and place it under this dev server directory. This jar file might be updated periodically.

## Prerequisites
```
pip install -r requirements.txt
```

## Running Dev Server
```
python -m flask run --port 9000 --host 0.0.0.0 --debug
```
