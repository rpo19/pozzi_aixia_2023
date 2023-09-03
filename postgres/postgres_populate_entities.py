import json
import argparse
import numpy as np
import base64
import pandas as pd
import psycopg
from tqdm import tqdm
import gzip
import itertools

max_title_len = 100
chunksize = 500

def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v

def ents_generator(args):
    indexer = args.indexer

    if args.entity_catalogue.endswith('gz'):
        fin = gzip.open(args.entity_catalogue, "rt")
    else:
        fin = open(args.entity_catalogue, "r")

    for local_idx in itertools.count():
        line = fin.readline()
        # if line is empty
        # end of file is reached
        if not line:
            break

        entity = json.loads(line)

        split = entity["idx"].split("curid=")
        if len(split) > 1:
            wikipedia_id = int(split[-1].strip())
        else:
            wikipedia_id = int(entity["idx"].strip())

        title = entity["title"]
        title = title[:max_title_len]
        text = entity["text"]

        yield local_idx, wikipedia_id, indexer, title, text

    fin.close()

def populate(connection, table_name):

    total = args.total

    with connection.cursor() as cursor:
        with cursor.copy("COPY {} (id, indexer, wikipedia_id, title, descr) FROM STDIN".format(table_name)) as copy:
            for id, wikipedia_id, indexer, title, text in tqdm(ents_generator(args), total=total):
                copy.write_row((id, indexer, wikipedia_id, title, text))
    connection.commit()
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # indexer
    parser.add_argument(
        "--postgres", type=str, default="", help="postgresql url (e.g. postgresql://username:password@localhost:5432/mydatabase)",
    )
    parser.add_argument(
        "--entity_catalogue",
        dest="entity_catalogue",
        type=str,
        # default="models/tac_entity.jsonl",  # TAC-KBP
        default="models/entity.jsonl",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )
    parser.add_argument(
        "--table-name",
        dest="table_name",
        type=str,
        default=None,  # ALL WIKIPEDIA!
        help="Postgres table name.",
    )
    parser.add_argument(
        "--indexer",
        dest="indexer",
        type=int,
        default=0,
        help="Indexer id.",
    )
    parser.add_argument(
        "--total",
        dest="total",
        type=int,
        default=-1,
        help="Total number of entities.",
    )

    args = parser.parse_args()

    assert args.table_name is not None, 'Error: table-name is required!'

    connection = psycopg.connect(args.postgres)

    print('Populating postgres...')
    populate(connection, args.table_name)

    connection.close()

