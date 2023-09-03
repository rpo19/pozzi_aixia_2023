import argparse
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import uvicorn
import numpy as np
import base64
from typing import List, Optional
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer
import psycopg
import os
from gatenlp import Document
from itertools import repeat
import requests
# from annoy import AnnoyIndex

class _Index:
    def __init__(self, n):
        self.ntotal = n
# class AnnoyWrapper:
#     def __init__(self, annoyIndex):
#         self._index = annoyIndex
#         self.index = _Index(self._index.get_n_items())
#         self.index_type = 'annoy'
#     def search_knn(self, encodings, top_k):
#         candidates = []
#         scores = []
#         for v in encodings:
#             _c, _s = self._index.get_nns_by_vector(v, top_k, include_distances=True)
#             candidates.append(_c)
#             scores.append(_s)
#         return scores, candidates
class HttpIndexer:
    # pass the index as http:example.com:13:r (omit http:// from the url)
    def __init__(self, url, only_indexes=None):
        if not url.startswith('http://'):
            url = 'http://' + url
        self.url = url
        self.only_indexes = only_indexes
        self.type = 'http'
        self.index = _Index(10) # dummy ntotal set to 10
    def search_knn(self, encodings, top_k):
        encodings = [vector_encode(e) for e in encodings]
        body = {
            'encodings': encodings,
            'top_k': top_k,
            'only_indexes': self.only_indexes,
        }
        res = requests.post(self.url + '/api/indexer/search', json=body)
        if res.ok:
            return res.json()
        else:
            print('Http error url', self.url)
            return None
    def id2info(self, body):
        body = dict(body)
        res = requests.post(self.url + '/api/indexer/info', json=body)
        return res.json()
        

def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v

class Input(BaseModel):
    encodings: List[str]
    top_k: int
    only_indexes: Optional[List[int]]

class Idinput(BaseModel):
    id: int
    indexer: int

indexes = {}
rw_index = None

def id2url(wikipedia_id):
    global language
    if wikipedia_id > 0:
        return "https://{}.wikipedia.org/wiki?curid={}".format(language, wikipedia_id)
    else:
        return ""

def id2props(wikipedia_id):
    global language
    url = 'https://{}.wikipedia.org/w/api.php?action=query&pageids={}&prop=extracts|pageimages&exchars=200&explaintext=true&pithumbsize=480&format=json'
    res = requests.get(url.format(language, wikipedia_id))
    if res.ok:
        resj = res.json()
        assert str(wikipedia_id) in resj['query']['pages']
        props = resj['query']['pages'][str(wikipedia_id)]
        return props
    else:
        return {}

app = FastAPI()

@app.post('/api/indexer/reset/rw')
async def reset():
    # reset rw index
    index_type = indexes[rw_index]['index_type']
    del indexes[rw_index]['indexer']
    if index_type == 'flat':
        indexes[rw_index]['indexer'] = DenseFlatIndexer(args.vector_size)
        indexes[rw_index]['indexer'].serialize(indexes[rw_index]['path'])
    else:
        raise Exception('Not implemented for index {}'.format(index_type))

    # reset db
    try:
        with dbconnection.cursor() as cur:
            print('deleting from db...')
            cur.execute("""
                DELETE
                FROM
                    entities
                WHERE
                    indexer = %s;
                """, (indexes[rw_index]['indexid'],))
        dbconnection.commit()

        return {'res': 'OK'}

    except BaseException as e:
        print('RESET query ERROR. Rolling back.')
        dbconnection.rollback()

        return {'res': 'ERROR'}


@app.post('/api/indexer/search/doc')
# remember `content-type: application/json`
async def search_from_doc_api(doc: dict = Body(...)):
    default_top_k = 10
    if doc.get('features', {}).get('top_k'):
        top_k = doc.get('features', {}).get('top_k')
    else:
        top_k = default_top_k
    return search_from_doc_topk(top_k, doc)

@app.post('/api/indexer/search/doc/{top_k}')
async def search_from_doc_topk_api(top_k: int, doc: dict = Body(...)):
    return search_from_doc_topk(top_k, doc)

def search_from_doc_topk(top_k, doc):
    doc = Document.from_dict(doc)

    annsets_to_link = set([doc.features.get('annsets_to_link', 'entities_spacy_v0.1.0')])

    encodings = []
    mentions = []
    for annset_name in set(doc.annset_names()).intersection(annsets_to_link):
        # if not annset_name.startswith('entities'):
        #     # considering only annotation sets of entities
        #     continue
        for mention in doc.annset(annset_name):
            if 'linking' in mention.features and mention.features['linking'].get('skip', False):
                # DATES should skip = true bcs linking useless
                continue
            enc = mention.features['linking']['encoding']
            encodings.append(enc)
            mentions.append(mention)

    all_candidates_4_sample_n = search(encodings, top_k)

    for mention, cands in zip(mentions, all_candidates_4_sample_n):
        # dummy is set when postgres is empty
        if len(cands) == 0 or ('dummy' in cands[0] and cands[0]['dummy'] == 1):
            mention.features['is_nil'] = True
        else:
            top_cand = cands[0]
            # TODO here for backward compatibility
            mention.features['linking']['top_candidate'] = top_cand
            mention.features['linking']['candidates'] = cands
            #
            mention.features['title'] = top_cand['title']
            mention.features['url'] = top_cand['url']
            mention.features['additional_candidates'] = cands

    if not 'pipeline' in doc.features:
        doc.features['pipeline'] = []
    doc.features['pipeline'].append('indexer')

    return doc.to_dict()

@app.post('/api/indexer/info')
async def id2info_api(idinput: Idinput):
    """
    input: (id, indexer)
    ouput: (id, indexer) -> info
    TODO continue
    """
    if not idinput.indexer in indexes:
        raise HTTPException(status_code=400, detail="Unknown indexer id.")

    if indexes[idinput.indexer]['index_type'] == 'http':
        return indexes[idinput.indexer]['indexer'].id2info(idinput)
    else:
        with dbconnection.cursor() as cur:
            cur.execute("""
                SELECT
                    id, indexer, title, wikipedia_id, type_, wikidata_qid, redirects_to, descr
                FROM
                    entities
                WHERE
                    id = %s AND
                    indexer = %s;
                """, (idinput.id, idinput.indexer))
            id2info = cur.fetchall()
        assert len(id2info) == 1
        x = id2info[0]
        return {
                    'id': x[0],
                    'indexer': x[1],
                    'title': x[2],
                    'wikipedia_id': x[3],
                    'type': x[4],
                    'wikidata_qid': x[5],
                    'redirects_to': x[6],
                    'descr': x[7],
                    'url': id2url(x[3]),
                    'props': id2props(x[3])
                }

@app.post('/api/indexer/search')
async def search_api(input_: Input):
    encodings = input_.encodings
    top_k = input_.top_k
    only_indexes = input_.only_indexes
    return search(encodings, top_k, only_indexes)

def search(encodings, top_k, only_indexes=None):
    encodings = np.array([vector_decode(e) for e in encodings])
    all_candidates_4_sample_n = []
    for i in range(len(encodings)):
        all_candidates_4_sample_n.append([])
    for index in indexes.values():
        if only_indexes and index['indexid'] not in only_indexes:
            # skipping index not in only_indexes
            continue
        indexer = index['indexer']
        if index['index_type'] != 'http':
            if indexer.index.ntotal == 0:
                scores = np.zeros((encodings.shape[0], top_k))
                candidates = -np.ones((encodings.shape[0], top_k)).astype(int)
            else:
                scores, candidates = indexer.search_knn(encodings, top_k)
            n = 0
            candidate_ids = set([id for cs in candidates for id in cs])

            try:
                with dbconnection.cursor() as cur:
                    cur.execute("""
                        SELECT
                            id, title, wikipedia_id, type_, wikidata_qid, redirects_to
                        FROM
                            entities
                        WHERE
                            id in ({}) AND
                            indexer = %s;
                        """.format(','.join([str(int(id)) for id in candidate_ids])), (index['indexid'],))
                    id2info = cur.fetchall()
            except BaseException as e:
                print('SELECT query ERROR. Rolling back.')
                dbconnection.rollback()

            id2info = dict(zip(map(lambda x:x[0], id2info), map(lambda x:x[1:], id2info)))
            for _scores, _cands, _enc in zip(scores, candidates, encodings):

                # for each samples
                for _score, _cand in zip(_scores, _cands):
                    raw_score = float(_score)
                    _cand = int(_cand)
                    if _cand == -1:
                        # -1 means no other candidates found
                        break
                    # # compute dot product always (and normalized dot product)

                    if _cand not in id2info:
                        # candidate removed from kb but not from index (requires to reconstruct the whole index)
                        all_candidates_4_sample_n[n].append({
                            'raw_score': -1000.0,
                            'id': _cand,
                            'wikipedia_id': 0,
                            'title': '',
                            'url': '',
                            'type_': '',
                            'indexer': index['indexid'],
                            'score': -1000.0,
                            'norm_score': -1000.0,
                            'dummy': 1
                        })
                        continue
                    title, wikipedia_id, type_, wikidata_qid, redirects_to = id2info[_cand]

                    if index['index_type'] == 'flat':
                        embedding = indexer.index.reconstruct(_cand)
                    elif index['index_type'] == 'hnsw':
                        embedding = indexer.index.reconstruct(_cand)[:-1]
                        _score = np.inner(_enc, embedding)
                    # elif index['index_type'] == 'annoy':
                    #     embedding = indexer._index.get_item_vector(_cand)
                    elif index['index_type'] == 'http':
                        # call another indexer instance
                        pass
                    else:
                        raise Exception('Should not happen.')

                    # normalized dot product
                    _enc_norm = np.linalg.norm(_enc)
                    _embedding_norm = np.linalg.norm(embedding)
                    _norm_factor = max(_enc_norm, _embedding_norm)**2
                    _norm_score = _score / _norm_factor

                    all_candidates_4_sample_n[n].append({
                            'raw_score': raw_score,
                            'id': _cand,
                            'wikipedia_id': wikipedia_id,
                            'wikidata_qid': wikidata_qid,
                            'redirects_to': redirects_to,
                            'title': title,
                            'url': id2url(wikipedia_id),
                            'type_': type_,
                            'indexer': index['indexid'],
                            'score': float(_score),
                            'norm_score': float(_norm_score)
                        })
        
                n += 1
        else:
            # indexer http
            all_candidates_4_sample_n_http = indexer.search_knn(encodings, top_k)
            if all_candidates_4_sample_n_http:
                assert len(all_candidates_4_sample_n_http) == len(all_candidates_4_sample_n)
                for i in range(len(all_candidates_4_sample_n)):
                    all_candidates_4_sample_n[i].extend(all_candidates_4_sample_n_http[i])
    # sort
    for _sample in all_candidates_4_sample_n:
        _sample.sort(key=lambda x: x['score'], reverse=True)
    return all_candidates_4_sample_n

class Item(BaseModel):
    encoding: str
    wikipedia_id: Optional[int]
    title: str
    descr: Optional[str]
    type_: Optional[str]

@app.post('/api/indexer/add/doc')
async def add_doc(doc: dict = Body(...)):
    doc = Document.from_dict(doc)

    if 'clusters' not in doc.features or not doc.features['clusters']:
        # emtpy list
        print('Nothing to add.')
        return doc.to_dict()

    # TODO refactor
    for name_clusters in doc.features['clusters']:
        for key, clusters in name_clusters.items():
            items = []

            for c in clusters:
                items.append(
                    Item(title=c['title'], encoding=c['center']))

            res_add = add(items)

            for c, id, indexer in zip(clusters, res_add['ids'], repeat(res_add['indexer'])):
                c['index_id'] = id
                c['index_indexer'] = indexer

            if not 'pipeline' in doc.features:
                doc.features['pipeline'] = []
            doc.features['pipeline'].append('indexer_add')

            # TODO now works with the first annotation set
            break
        # TODO
        break

    return doc.to_dict()

@app.post('/api/indexer/add')
async def add_api(items: List[Item]):
    return add(items)

def add(items: List[Item]):
    if rw_index is None:
        raise HTTPException(status_code=404, detail="No rw index!")

    # input: embeddings --> faiss
    # --> postgres
    # wikipedia_id ?
    # title
    # descr ?
    # embedding

    indexer = indexes[rw_index]['indexer']
    indexid = indexes[rw_index]['indexid']
    indexpath = indexes[rw_index]['path']

    # add to index
    embeddings = [vector_decode(e.encoding) for e in items]
    embeddings = np.stack(embeddings).astype('float32')
    indexer.index_data(embeddings)
    ids = list(range(indexer.index.ntotal - embeddings.shape[0], indexer.index.ntotal))
    # save index
    print(f'Saving index {indexid} to disk...')
    indexer.serialize(indexpath)

    global args


    # add to postgres
    try:
        with dbconnection.cursor() as cursor:
            with cursor.copy("COPY entities (id, indexer, wikipedia_id, title, descr, type_) FROM STDIN") as copy:
                for id, item in zip(ids, items):
                    wikipedia_id = -1 if item.wikipedia_id is None else item.wikipedia_id
                    copy.write_row((id, indexid, wikipedia_id, item.title[:args.title_max_len], item.descr, item.type_))
        dbconnection.commit()

        return {
            'res': 'OK',
            'ids': ids,
            'indexer': indexid
        }
    except BaseException as e:
        print('ADD query ERROR. Rolling back.')
        dbconnection.rollback()

        raise HTTPException(status_code=500, detail="ADD query ERROR. Rolling back.")

def load_models(args):
    assert args.index is not None, 'Error! Index is required.'
    for index in args.index.split(','):
        index_type, index_path, indexid, rorw = index.split('+')
        print('Loading {} index from {}, mode: {}...'.format(index_type, index_path, rorw))
        if os.path.isfile(index_path):
            if index_type == "flat":
                indexer = DenseFlatIndexer(1)
                indexer.deserialize_from(index_path)
            elif index_type == "hnsw":
                indexer = DenseHNSWFlatIndexer(1)
                indexer.deserialize_from(index_path)
            # elif index_type == 'annoy':
            #     _annoy_idx = AnnoyIndex(args.vector_size, 'dot')
            #     _annoy_idx.load(index_path)
            #     indexer = AnnoyWrapper(_annoy_idx)
            else:
                raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
        else:
            if index_type == "flat":
                indexer = DenseFlatIndexer(args.vector_size)
            elif index_type == "hnsw":
                raise ValueError("Error! HNSW index File not Found! Cannot create a hnsw index from scratch.")
            elif index_type == 'http':
                indexer = HttpIndexer(index_path, [indexid])
            else:
                raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
        indexes[int(indexid)] = {
            'indexer': indexer,
            'indexid': int(indexid),
            'path': index_path,
            'index_type': index_type
            }

        global rw_index
        if rorw == 'rw':
            assert rw_index is None, 'Error! Only one rw index is accepted.'
            rw_index = int(indexid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # indexer
    parser.add_argument(
        "--index", type=str, default=None, help="comma separate list of paths to load indexes [type:path:indexid:ro/rw] (e.g: hnsw:index.pkl:0:ro,flat:index2.pkl:1:rw)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="host to listen at",
    )
    parser.add_argument(
        "--port", type=int, default="30301", help="port to listen at",
    )
    parser.add_argument(
        "--postgres", type=str, default=None, help="postgres url (e.g. postgres://user:password@localhost:5432/database)",
    )
    parser.add_argument(
        "--vector-size", type=int, default="1024", help="The size of the vectors", dest="vector_size",
    )
    parser.add_argument(
        "--title-max-len", type=int, default=100, help="Max title len", dest="title_max_len",
    )
    parser.add_argument(
        "--language", type=str, default="en", help="Wikipedia language (en,it,...).",
    )

    args = parser.parse_args()

    assert args.postgres is not None, 'Error. postgres url is required.'
    dbconnection = psycopg.connect(args.postgres)

    language = args.language

    print('Loading indexes...')
    load_models(args)
    print('Loading complete.')

    uvicorn.run(app, host = args.host, port = args.port)
    dbconnection.close()
