# hnrq-db

This is a repository for the setup of multiple dockerized database instances for HNRQ project in general (mostly for I-Rex development for now).

## Setup Procedures

1. Make sure you have Docker. It is recommended that you add [Docker to your sudo group](https://docs.docker.com/engine/install/linux-postinstall/) to avoid typing `sudo` for all Docker commands.
2. Run `docker compose up -d --build`. Or you can split this into two steps: first run `docker compose build`, then `docker compose up -d`.
3. Once everything is up and running, go into the container to build the bloom filter extension. The following command will give you an interactive shell from the container:
```
docker compose exec -u hnrq -it postgres bash --login
```
4. Run `cd ~/shared/`, and run the following commands to build and install the extension:
```
bash install.sh
```

Note: it turns out that `sudo` command is not allowed in docker entrypoint, and this is why we cannot install extension at startup. Maybe we can find some hacks later on but now I have simplify the extension installation with step 4. 

## Summary
The entire procedure is not so automated right now. We can try to improve in the future once we are ready to package everything.
