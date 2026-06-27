# Sargable Summary Extension for PostgreSQL

`blmfl` is a sargable summary extension for PostgreSQL that uses a bloom 
filter.  It defines acustom aggregation function that builds a summary for 
a collection ofvalues for a column, such that the summary information can 
be used tooptimize access to rows whose column value falls within the 
summarized collection.

## Setting up Environment

* Download and install [Docker
  Desktop](https://docs.docker.com/get-docker/) on your laptop.  If
  you have a previous installation, perform an update to make sure you
  stay on the latest version.

* You should have some command-line interface running on your laptop.
  For Mac/Linux, this is the Terminal program, and for Windows, this
  can be the PowerShell.  We will call this your "host shell".

* Pick a directory for this project, e.g., `~/blmfl` (for Mac/Linux)
  or `C:\blmfl` (for Windows).  We will call this directory your
  "project directory (on laptop)".  WARNING: We do NOT recommend
  putting this directory in cloud storage --- permissions on such
  files may be messed up and you may have trouble running programs
  that depend on having specific permissions.

* Make a copy of the file `env-template.txt` in the container subdirectory and
  name it `.env`; then edit the fields to your liking.  The following commands
  assume you are using Mac/Linux (replace the path argument to the `cd`
  command below with what you had chosen as your project directory):
  ```
  cd ~/blmfl
  cp env-template.txt .env
  nano .env
  ```
  For all subsequent steps, we assume that Docker Desktop is up and
  running, and you are executing commands in your host shell with your
  working directory being the container subdirectory (where
  `docker-compose.yaml` resides).

* Use the following to prepare for launching containers the first time:
  ```
  docker compose build
  ```
  This will take quite some time, as it involves downloading various
  software and setting things up, but you only have to run this once.

* We assume that for convenience, you will want to access your entire
  project directory from within your container as well.  Hence, before
  we launch the containers, open Settings for Docker Desktop, go to
  Resources -> File Sharing, and add your project directory to the
  list (to be "bind mounted").

* To start the containers for the first time, 
  ```
  docker compose up -d
  ```

* While the containers are running, to log into your container running
  the PostgreSQL server, type (replace `your_login` with whatever you
  chose earlier):
  ```
  docker compose exec -u your_login -it postgres bash --login
  ```
  You will be within what we would call your "project shell."  WARNING:
  Don't forget the `--login` switch; without it, your environment
  won't be properly initialized.

  Once you are in, you can also access your entire project directory on
  the host through `~/shared/` in your container.  You will have full
  read-write access.  WARNING: Files outside this directory reside
  ONLY within your container; unless you back them up in a different
  way, don't expect them to be around forever!

  HINT: When everything is command-line, sometimes it is hard to tell
  whether you are inside your host shell, your project shell, or
  even some command-line interface running within your project
  shell.  Pay attention to the command-line prompt, and make sure you
  issue the right commands in the right environment.

* Following your initial setup, assuming that Docker Desktop is up and
  running, and you are in your host shell with your working directory
  being the project directory (where `docker-compose.yaml` resides),
  you can start/stop running the containers using:
  ```
  docker compose start
  docker compose stop
  ```

## Building, Installing, and Testing PostgreSQL Extension

* While in your project shell (on your container) in the project
  directory, issue the following command to compile and build the
  extension:
  ```
  make clean
  make
  ```

* Assuming all went well, use the following command to install the
  extension on the PostgreSQL server:
  ```
  sudo make install
  ```

* Then, you can test the extension on a PostgreSQL database.  The file
  `sql/blmfl_test.sql` is a very simple test case.  To run it (on
  the default database):
  ```
  psql -f sql/blmfl_test.sql
  ```
  Read the file to see the syntax for enabling the extension on a
  database and for using the extension.

## Directory Structure

* `blmfl.c`: This is the meat of the C implementation.

* `blmfl--0.0.1.sql`: This is the SQL installation script that tell
  PostgreSQL where to find the C functions implementing various
  functionalities of the extension.

* `blmfl.control`: Meta data for the extension.  No need to modify
  in most cases.

* `Makefile`: No need to modify in most cases.

* `sql/`: Testing scripts.  There is a more elaborate system for
  adding regression tests for PostgreSQL extensions, but we are
  ignoring that for now.

## Design Notes and Useful Pointers

* PostgreSQL resources:
  - https://www.postgresql.org/docs/current/sql-createaggregate.html
  - https://www.postgresql.org/docs/current/xfunc-c.html
  - https://www.postgresql.org/docs/current/xaggr.html
  - https://doxygen.postgresql.org/arrayfuncs_8c.html

* Much inspiration for the current skeleton implementation came from
  the HyperLogLog PostgreSQL extension:
  https://github.com/citusdata/postgresql-hll/tree/master
  Here, the internal aggregation state is essentially a C structure
  (of type `INTERNAL`); only when we finalize the result, we convert
  the state to an array of integers.
  - We should definitely study this HyperLogLog implementation further
    to get more optimization ideas.

* Alternative approaches are possible, e.g., keep the state as a
  PostgreSQL array, in the style of the following extension:
  https://github.com/pjungwir/aggs_for_arrays/tree/master
  Drawback is that PostgreSQL array API is extremely annoying.
