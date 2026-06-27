#!/bin/bash

# mypath=`realpath $0`
# mybase=`dirname $mypath`
mybase=/home/$POSTGRES_USER/shared

# declare all database, create tables and load data
dbnames=(beers_db0 beers_db1 tpch_db0 tpch_db1)

# tmp create irex db
dropdb -U $POSTGRES_USER irex
createdb -U $POSTGRES_USER irex

for dbname in ${dbnames[@]}; do
if [[ -n `psql -lqt | cut -d \| -f 1 | grep -w "$dbname"` ]]; then
    dropdb -U $POSTGRES_USER $dbname
fi
createdb -U $POSTGRES_USER $dbname --encoding=UTF-8 --locale=C --template=template0

cd $mybase
cd $dbname
psql -U $POSTGRES_USER -af create.sql $dbname
psql -U $POSTGRES_USER -af load.sql $dbname
psql -U $POSTGRES_USER -af $mybase/helper.sql $dbname
cd ..
done

# cd $mybase/sargsum
# make clean
# make
# sudo make install
# for dbname in ${dbnames[@]}; do
#     psql -U $POSTGRES_USER -d $dbname -c "CREATE EXTENSION BLMFL;"
# done
