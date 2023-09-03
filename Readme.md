# Setup

- Install docker and docker compose; https://docs.docker.com/engine/install/
- Copy env and set variables
```
cp env-sample.txt .env
```
- Edit `.env`; you should set at least `LOCAL_WORKSPACE_FOLDER` to the folder containing your project.
- Download and extract `models.zip` from https://drive.google.com/drive/folders/1eH03MjHkgHiFS1E12XF8ku8VIzz13d9B?usp=sharing
Once extracted you should see:
```
> ls
biencoder
docker-compose.yml
env-sample.txt
indexer
models
nilpredictor
postgres
Readme.md
spacyner
```
- Run `docker-compose up -d` (you may need sudo)
- Import data with `docker-compose exec -T postgres pg_restore -U postgres -a -Fc -d postgres < models/pg_dump_aixia_2023.dump`; you may need sudo.
- Create a virtualenv (suggested); see https://virtualenv.pypa.io/en/latest/
- Install the requirements
```
# we suggest to run it in a virtualenv
pip install -r requirements.txt
```
- Start jupyter and run the notebook
```
jupyter-notebook --port 9010
```
- Open the link provided by jupyter

- Try the pipeline

