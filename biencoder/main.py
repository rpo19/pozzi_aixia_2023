import argparse
from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn
from blink.main_dense import load_biencoder, _process_biencoder_dataloader
from blink.biencoder.eval_biencoder import get_candidate_pool_tensor
from typing import List, Optional
import json
from tqdm import tqdm
import torch
import numpy as np
import base64
import logging
from torch.utils.data import DataLoader, SequentialSampler
from gatenlp import Document

def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v

class Mention(BaseModel):
    label = 'unknown'
    label_id = -1
    context_left: str
    context_right:str
    mention: str
    start_pos: Optional[int]
    end_pos: Optional[int]
    sent_idx: Optional[int]

class Entity(BaseModel):
    title: str
    descr: str

app = FastAPI()

@app.post('/api/blink/biencoder/mention/doc')
# remember `content-type: application/json`
async def encode_mention_from_doc(doc: dict = Body(...)):
    doc = Document.from_dict(doc)

    annsets_to_link = set([doc.features.get('annsets_to_link', 'entities_merged')])

    samples = []
    mentions = []

    for annset_name in set(doc.annset_names()).intersection(annsets_to_link):
        # if not annset_name.startswith('entities'):
        #     # considering only annotation sets of entities
        #     continue
        for mention in doc.annset(annset_name):
            if 'linking' in mention.features and mention.features['linking'].get('skip', False):
                # DATES should skip = true bcs linking useless
                continue
            blink_dict = {
                # TODO use sentence instead of document?
                # TODO test with very big context
                'context_left': mention.features['context_left'] if 'context_left' in mention.features \
                                                                    else doc.text[:mention.start],
                'context_right': mention.features['context_right'] if 'context_right' in mention.features \
                                                                    else doc.text[mention.end:],
                'mention': mention.features['mention'] if 'mention' in mention.features \
                                                                    else doc.text[mention.start:mention.end],
                #
                'label': 'unknown',
                'label_id': -1,
            }
            samples.append(blink_dict)
            mentions.append(mention)

    dataloader = _process_biencoder_dataloader(
        samples, biencoder.tokenizer, biencoder_params
    )
    encodings = _run_biencoder_mention(biencoder, dataloader)
    if len(encodings) > 0:
        assert encodings[0].dtype == 'float32'
    encodings = [vector_encode(e) for e in encodings]

    for mention, enc in zip(mentions, encodings):
        mention.features['linking'] = {
            'encoding': enc,
            'source': 'blink_biencoder'
        }

    if not 'pipeline' in doc.features:
        doc.features['pipeline'] = []
    doc.features['pipeline'].append('biencoder')

    return doc.to_dict()

@app.post('/api/blink/biencoder/mention')
async def encode_mention(samples: List[Mention]):
    samples = [dict(s) for s in samples]
    dataloader = _process_biencoder_dataloader(
        samples, biencoder.tokenizer, biencoder_params
    )
    encodings = _run_biencoder_mention(biencoder, dataloader)
    if len(encodings) > 0:
        assert encodings[0].dtype == 'float32'
    #assert np.array_equal(encodings[0], vector_decode(vector_encode(encodings[0]), np.float32))
    ## dtype float32
    encodings = [vector_encode(e) for e in encodings]
    return {'samples': samples, 'encodings': encodings}

@app.post('/api/blink/biencoder/entity')
async def encode_entity(samples: List[Entity]):
    # entity_desc_list: list of tuples (title, text)
    entity_desc_list = [(s.title, s.descr) for s in samples]
    candidate_pool = get_candidate_pool_tensor(
        entity_desc_list,
        biencoder.tokenizer,
        biencoder_params["max_cand_length"],
        logger
    )
    sampler = SequentialSampler(candidate_pool)
    dataloader = DataLoader(
        candidate_pool, sampler=sampler, batch_size=8
    )

    encodings = _run_biencoder_entity(biencoder, dataloader)

    if len(encodings) > 0:
        assert encodings[0].dtype == 'float32'
    #assert np.array_equal(encodings[0], vector_decode(vector_encode(encodings[0]), np.float32))
    ## dtype float32
    encodings = [vector_encode(e) for e in encodings]
    return {'samples': samples, 'encodings': encodings}

def _run_biencoder_mention(biencoder, dataloader):
    biencoder.model.eval()
    encodings = []
    for batch in tqdm(dataloader):
        context_input, _, _ = batch
        with torch.no_grad():
            context_input = context_input.to(biencoder.device)
            context_encoding = biencoder.encode_context(context_input).numpy()
            context_encoding = np.ascontiguousarray(context_encoding)
        encodings.extend(context_encoding)
    return encodings

def _run_biencoder_entity(biencoder, dataloader):
    biencoder.model.eval()
    cand_encode_list = []
    for batch in tqdm(dataloader):
        cands = batch
        cands = cands.to(biencoder.device)
        with torch.no_grad():
            cand_encode = biencoder.encode_candidate(cands).numpy()
            cand_encode = np.ascontiguousarray(cand_encode)
        cand_encode_list.extend(cand_encode)
    return cand_encode_list

def load_models(args):
    # load biencoder model
    with open(args.biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = args.biencoder_model
    biencoder = load_biencoder(biencoder_params)
    return biencoder, biencoder_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # biencoder
    parser.add_argument(
        "--biencoder_model",
        dest="biencoder_model",
        type=str,
        default="models/biencoder_wiki_large.bin",
        help="Path to the biencoder model.",
    )
    parser.add_argument(
        "--biencoder_config",
        dest="biencoder_config",
        type=str,
        default="models/biencoder_wiki_large.json",
        help="Path to the biencoder configuration.",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )
    parser.add_argument(
        "--port", type=int, default="30300", help="port to listen at",
    )

    args = parser.parse_args()

    logger = logging.getLogger('biencoder_micros')

    print('Loading biencoder...')
    biencoder, biencoder_params = load_models(args)
    print('Device:', biencoder.device)
    print('Loading complete.')

    uvicorn.run(app, host = args.host, port = args.port)
