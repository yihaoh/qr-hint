#!/bin/bash

######################################################################
# Basics:
apt-get -qq update
DEBIAN_FRONTEND=noninteractive apt-get -yq -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" upgrade
apt-get -qq --yes install wget nano vim less gnupg curl man coreutils git

######################################################################
# Python 3.11:
apt-get -qq --yes install python3.11
update-alternatives --install /usr/bin/python python /usr/bin/python3.11 3
apt-get -qq --yes install python3-pip
apt-get -qq --yes install python-is-python3
python -m pip install --break-system-packages poetry

######################################################################
# PostgreSQL client and development libraries:
apt-get -qq --yes install postgresql-client-16 libpq-dev postgresql-server-dev-16
python -m pip install --break-system-packages psycopg2-binary SQLAlchemy