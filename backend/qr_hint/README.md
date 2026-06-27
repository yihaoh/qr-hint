# Qr-Hint Implementation

## Overview
The project heavily involves testing implications/equivalence between boolean formula, and such testing is done through the [z3 library](https://pypi.org/project/z3-solver/), a tutorial can be found [here](https://ericpony.github.io/z3py-tutorial/guide-examples.htm).

### Main Python Files
`query_info.py`: Parsing queries into test-ready format. It leverages [this query parser](https://gitlab.cs.duke.edu/junyang/irex), and you can download it [here](https://www.cs.duke.edu/~junyang/tmp/irex/sqlanalyzer.jar). Make sure `sqlanalyzer.jar` is in the `qr_hint` folder.

`query_test.py`: Main entry of the project. It collects query info and perform stage testing.

`boolean_parse_tree.py`: Boolean syntax tree that is built on top of the xtree for the convenience of testing WHERE and HAVING.

### Helper Python Files
`fix_generator.py`: our baseline fix derivation.

`fix_optimizer.py`: our optimized fix derivation (slower but smaller fixes).

`subtree_iter.py`: defines a iterator that iterates through all combinations of repair sites.

`global_var_*.py`: some config needed by the `QueryInfo` object.

`utils.py`: some general helper functions.


## How to run for development
1. Make sure the Postgres is up and running either on your localhost or through Docker (if using Docker, make sure its port 5432 is bridged to port 5432 on the localhost). For required database instances, feel free to create them accordingly based on the information in `global_var_*.py`. The Dockerized database under `../db` contains a toy beer database, which is probably sufficient for development. Note that we do not need the database to contain actual data, having the schema is enough.
2. Make sure you have Java JDK 11 or newer.
3. Now feel free to explore `testing.ipynb`, and make changes to other files accordingly.

