# qr-hint

This repository contain the source code for qr-hint, as well as a full version of the paper (same main body with extra appendix).

## Note
1. There is a typo in Lemma 5.1 in the submitted version of the paper (the viability check should be equivalence between two formula as stated at the beginning of section 5). This typo is corrected in the full version under this repository. We sincerely apologize for any inconvenience and confusion.
2. Please do not look into `global_var_beers.py`, `global_var_dblp.py` and `global_var_tpc.py` as they are importing a Java library developed by the authors, andh the Java package contains information that might leak authors' identity.
3. The Quine-McCluskey implementations are borrowed from this [GitHub repo](https://github.com/Kumbong/QuineMcCluskey) as well as this [repo](https://github.com/prekageo/optistate/tree/master). The maintainers of these repositories are not affliated with the authors of the paper in any way.

## Setup
To run the demo code, please make sure you have a running PostgreSQL database with a user named "postgres" (should be default) and password "postgres". If you are using Docker, please feel free to check out what we have prepared under `db` directory. It is a classical beers database instance (same one as the running example in the paper).

Some system pre-requisites:
1. Python 3.8 or above, remember to install all libraries in `requirements.txt`.
2. Java JDK 11 or later, as we are using Apache Calcite.
3. Jupyter notebook for demo.

## Demo
Feel free to open Testing.ipynb with Jupyter Notebook to see the demo example. You can also play with it a little more. Note that when you write queries, make sure the first letter of all table names must be capitalized, otherwise it might cause errors in the program.

The most robust or bug-free part is testing WHERE, in which we try to cover a lot of corner cases and it is the most important component. 

If you would like to see the experiments scripts, they are under `qr-hint-code/tpc-h-test`. `num-pred-test` corresponds to the runtime test for number of predicates, and `num-rs-test` corresponds to the accuracy test where multiple errors are injected. If you would like to run it yourself, make sure you do the following:
1. In PostgreSQL, create a database called `tpc`.
2. Create tables using the commands in `tpc-h-test/create.sql`.
3. In `query_info.py`, comment out the import of `global_var_beers` and comment in the import of `global_var_tpc`.







