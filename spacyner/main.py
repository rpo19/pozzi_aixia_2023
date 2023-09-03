import argparse
from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn
import spacy
from spacy.cli import download as spacy_download

from gatenlp import Document

DEFAULT_TAG='aplha_v0.1.0_spacy'

class Item(BaseModel):
    text: str

app = FastAPI()

def restructure_newline(text):
  return text.replace('\n', ' ')

@app.post('/api/spacyner')
async def encode_mention(doc: dict = Body(...)):

    # replace wrong newlines
    text = restructure_newline(doc['text'])

    doc = Document.from_dict(doc)
    entity_set = doc.annset('entities_{}'.format(args.tag))

    spacy_out = spacy_pipeline(text)

    # sentences
    if args.sents:
        sentence_set = doc.annset('sentences_{}'.format(args.tag))
        for sent in spacy_out.sents:
            # TODO keep track of entities in this sentence?
            sentence_set.add(sent.start_char, sent.end_char, "sentence", {
                "source": "spacy",
                "spacy_model":args.model
            })

    for ent in spacy_out.ents:
        # TODO keep track of sentences
        # sentence_set.overlapping(ent.start_char, ent.end_char)
        feat_to_add = {
            "ner": {
                "type": ent.label_,
                "score": 1.0,
                "source": "spacy",
                "spacy_model": args.model
                }}
        if ent.label_ == 'DATE':
            feat_to_add['linking'] = {
                "skip": True
            }

        entity_set.add(ent.start_char, ent.end_char, ent.label_, feat_to_add)

    if not 'pipeline' in doc.features:
        doc.features['pipeline'] = []
    doc.features['pipeline'].append('spacyner')

    return doc.to_dict()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )
    parser.add_argument(
        "--port", type=int, default="30304", help="port to listen at",
    )
    parser.add_argument(
        "--model", type=str, default="en_core_web_sm", help="spacy model to load",
    )
    parser.add_argument(
        "--tag", type=str, default=DEFAULT_TAG, help="AnnotationSet tag",
    )
    parser.add_argument(
        "--sents", action='store_true', default=False, help="Do sentence tokenization",
    )

    args = parser.parse_args()

    print('Loading spacy model...')
    # Load spacy model
    try:
        spacy_pipeline = spacy.load(args.model, exclude=['tok2vec', 'morphologizer', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
    except Exception as e:
        print('Caught exception:', e, '... Trying to download spacy model ...')
        spacy_download(args.model)
        spacy_pipeline = spacy.load(args.model, exclude=['tok2vec', 'morphologizer', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
    # sentences
    if args.sents:
        spacy_pipeline.enable_pipe('senter')

    print('Loading complete.')

    uvicorn.run(app, host = args.host, port = args.port)
