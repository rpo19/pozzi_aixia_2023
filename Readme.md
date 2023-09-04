# Named Entity Recognition and Linking for Entity Extraction from Italian Civil Judgements

Riccardo Pozzi, Riccardo Rubini, Christian Bernasconi, Matteo Palmonari

University of Milano-Bicocca, Milan, Italy

## Setup and Run the pipeline

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
models
models/spacy_it_trf_wikiner			        # spacy model
models/faiss_hnsw_ita_index.pkl			    # dense retrieval index for entities 
                                            # (see BLINK paper)
models/nilp_bi_max_secondiff_model.pickle	# NIL prediction model (features:
                                            # top linking score, top score - second
models/blink_biencoder_base_wikipedia_ita	# BLINK biencoder Italian
models/pg_dump_aixia_2023.dump			    # postgres db dump
                                            # contains entities (it.wikipedia.org)
models/bert-base-italian-xxl-uncased		# BERT Italian model
biencoder
docker-compose.yml
env-sample.txt
indexer
nilpredictor
postgres
Readme.md
spacyner
```
- Run (you may need sudo)
```
docker-compose up -d
```
- Import data (you may need sudo)
```
# executes pg_restore inside the postgres docker container
docker-compose exec -T postgres pg_restore -U postgres -a -Fc -d postgres < models/pg_dump_aixia_2023.dump
```
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

## Example NER training

- Prepare training data in spacy format, e.g., with train.spacy, dev.spacy, and test.spacy in the folder data.
- Prepare the `config.cfg` (see the example in this repo)
- Train spacy
```
spacy train config.cfg --output spacyner --gpu-id 0 --paths.train ./data/train.spacy --paths.dev ./data/valid.spacy
```
- Eval model
```
spacy evaluate --gpu-id 0 spacyner ./data/test.spacy
```
