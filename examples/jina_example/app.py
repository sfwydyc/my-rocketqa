__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import sys

import click
from jina import Flow
from jina import Document, DocumentArray, Flow
from jina.logging.logger import JinaLogger
from jina.clients.helper import pprint_routes


def index(file_name):
    f = Flow().load_config('flows/index.yml')
    logger = JinaLogger('app_index')
    with f:
        cnt = 0
        docs = DocumentArray()
        for line in open(file_name):
            doc = Document(id=cnt, uri=file_name)
            title, para = line.strip().split('\t')
            doc.text = line.strip()
            doc.tags['title'] = title
            doc.tags['para'] = para
            cnt += 1
            docs.append(doc)
        f.post(on='/index', inputs=docs)
        #f.post(on='/dump', parameters={'dump_path': './workspace/dump_lmdb', 'shards': 1})
        print(f'docs: {len(docs)}')


def query():

    f = Flow().load_config('flows/query.yml')
    with f:
        #while True:
        #    text = input('please type a query: ')
        #    if not text:
        #        break
        resp = f.post(on='/search', inputs=[Document(text="what is paula deen's brother")], return_results=True)


def query_restful(return_flow=False):
    f = Flow().load_config('flows/query.yml')
    f.use_rest_gateway()
    if return_flow:
        return f
    with f:
        f.block()


# @click.command()
# @click.option(
#     '--task',
#     '-t',
#     type=click.Choice(['index', 'index_incremental', 'query', 'query_restful'], case_sensitive=False),
# )
# @click.option('--num_docs', '-n', default=MAX_DOCS)
# @click.option('--top_k', '-k', default=5)

def main():
    #index('../marco.tp.1k')
    query()

if __name__ == '__main__':
    main()
    # main()
