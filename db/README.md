This directory contains what you need to run PostgreSQL (database server) together with pgAdmin (a web-based admin frontend to PostgreSQL) using Docker containers.

In this document:
- *host* refers to the personal laptop/workstation that you are working on;
- *dbhome* refers to the directory on host where this `README.md` file resides (together with `docker-compose.yaml`, etc.).

# Prerequsites

- Install [Docker](https://docs.docker.com/get-docker/) on your *host*.

# Setting Up

- Go into *dbhome* on your *host* and copy the template secret files:
  ```
  cp secrets/templates/* secrets/
  ```
  Then, using a plain-text editor:
  + edit the contents of `secrets/POSTGRES_USER` to the desired user (default `postgres` user is fine unless you like extra protection);
  + edit the contents of `secrets/POSTGRES_PASSWORD` to the desired password (**do NOT leave the default unchanged!**);
  + edit the contents of `secrets/pgadmin.env` to specify email and password for pgAdmin.
    * The default password here is only used upon the *initial* login to pgAdmin; you should **change the password once inside pgAdmin** (there is no need to update this file afterwards).

- Inside *dbhome* on your *host*, issue the following command to build and start the services for the first time:
  ```
  docker-compose up -d
  ```

# Using the Containers

- To start/stop running the services, issue the following command inside *dbhome* on your *host*:
  ```
  docker-compose start
  docker-compose stop
  ```

- Inside *dbhome* on your *host*, the directory `shared/` is also visible to the PostgreSQL server at `/shared/`.
  Any changes in this directory you made on your *host* will be reflected on the PostgreSQL server, and vice versa.
  `shared/db-beers/` contains scripts and data files to set up a sample database.

- To get `bash` shell access to the container running PostgreSQL, issue the following command inside *dbhome* on your *host*:
  ```
  docker-compose exec -u $(cat secrets/POSTGRES_USER) -w /shared postgres bash
  ```
  Once inside this `bash` shell, you can, for example, set up a sample database called `beers` from `shared/db-beers/` and exit:
  ```
  /shared/db-beers/setup.sh
  exit
  ```

- To get `psql` (SQL interpreter) access to the PostgreSQL server, issue the following command inside *dbhome* on your *host* (assuming you have already created the `beers` sample database):
  ```
  docker-compose exec -u $(cat secrets/POSTGRES_USER) -w /shared postgres psql beers
  ```

- To access the web-based pgAdmin, open a browser on your *host* and point it to `http://localhost:8080/`.
  You will need to use "Add New Server" to tell phAdmin how to connect to your PostgreSQL server:
  + give the server any name you like;
  + under "Connection: Host name/address", specify `postgres-server`;
  + under "Connection: User name", specify the same user name as in `secrets/POSTGRES_USER`.

- We have automatically created two named, persistent Docker volumes, `db_postgres-data` and `db_pgadmin-data`, for storage.
  Even if you destroy (`docker-compose down`) or rebuild (`docker-compose up -d`) the containers, these volumes will continue to hold your data and newly created containers will have access to them.
  If you want to list/destroy these volumes, use Docker volume management commands (`docker volume help`).