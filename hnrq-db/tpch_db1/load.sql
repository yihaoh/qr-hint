\copy "region"     from 'data/region.tbl'        DELIMITER '|' CSV;
\copy "nation"     from 'data/nation.tbl'        DELIMITER '|' CSV;
\copy "customer"   from 'data/customer.tbl'    DELIMITER '|' CSV;
\copy "supplier"   from 'data/supplier.tbl'    DELIMITER '|' CSV;
\copy "part"       from 'data/part.tbl'            DELIMITER '|' CSV;
\copy "partsupp"   from 'data/partsupp.tbl'    DELIMITER '|' CSV;
\copy "orders"     from 'data/orders.tbl'        DELIMITER '|' CSV;
\copy "lineitem"   from 'data/lineitem.tbl'    DELIMITER '|' CSV;