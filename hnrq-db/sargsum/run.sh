#!/bin/bash

make clean
make

sudo make install

psql -f sql/blmfl_test.sql