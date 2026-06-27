#!/bin/bash

DB=tpch
SCALE=1
DATADIR=/tmp/tpch-data/

make;

dropdb --if-exists $DB;
createdb $DB;
./dbgen -s $SCALE -f;

for i in `ls *.tbl`; do
    sed 's/|$//' $i > ${i/tbl/csv};
    echo $i;
done;

mkdir -p $DATADIR;
mv *.csv $DATADIR;
psql -d tpch -f tpch-build-db.sql;
