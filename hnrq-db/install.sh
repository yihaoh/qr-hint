#!/bin/bash

mybase=/home/$POSTGRES_USER/shared

# declare all database, create tables and load data
dbnames=(beers_db0 beers_db1 tpch_db0 tpch_db1)

cd $mybase/sargsum
make clean
make
sudo make install
for dbname in ${dbnames[@]}; do
    psql -U $POSTGRES_USER -d $dbname -c "CREATE EXTENSION BLMFL;"
done
