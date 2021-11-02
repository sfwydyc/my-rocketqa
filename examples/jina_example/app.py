__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import sys

import click
from jina import Flow
#from jina import TimeContext
#from jina import default_logger as logger

import cv2
import faiss
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc
import paddle
#paddle.set_device('cpu')
paddle.enable_static()


def load_config():

    """
    os.environ['JINA_PARA_FILE'] = os.environ.get('JINA_PARA_FILE', 'data/marco_test_para.txt')
    os.environ['JINA_WORKSPACE'] = os.environ.get('JINA_WORKSPACE', 'workspace')
    os.environ['JINA_PORT'] = os.environ.get('JINA_PORT', str(45678))
    os.environ['JINA_INDEX_BATCH_SIZE'] = os.environ.get('JINA_INDEX_BATCH_SIZE','256')
    os.environ['JINA_PREDICT_BATCH_SIZE'] = os.environ.get('JINA_PREDICT_BATCH_SIZE','1')
    """

def print_topk(resp, sentence):
    for d in resp.search.docs:
        print(f'Ta-Dah🔮, here are what we found for: {sentence}')
        for idx, match in enumerate(d.matches):

            score = match.score.value
            # if score < 0.0:
            #     continue
            print(f'> {idx:>2d}({score:.2f}). {match.text}')


def _index(f, data_fn, num_docs):
    with f:
        f.logger.info(f'Indexing {os.environ[data_fn]}')
        data_path = os.path.join(os.path.dirname(__file__), os.environ.get(data_fn, None))
        num_docs = min(num_docs, len(open(data_path).readlines()))
        index_batch_size = int(os.environ['JINA_INDEX_BATCH_SIZE'])
        with TimeContext(f'QPS: indexing {num_docs}', logger=f.logger):
            f.index_lines(filepath=data_path, batch_size=index_batch_size, read_mode='r', size=num_docs)


def index(num_docs):
    f = Flow().load_config('flows/index.yml')
    _index(f, 'JINA_PARA_FILE', num_docs)


def query(top_k):
    def ppr(x):
        print_topk(x, text)

    f = Flow().load_config('flows/query.yml')
    with f:
        while True:
            text = input('please type a query: ')
            if not text:
                break
            f.search_lines(
                lines=[text,],
                line_format='text',
                on_done=ppr,
                top_k=top_k
            )


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

def main(task, num_docs, top_k):
    config()
    """
    workspace = os.environ['JINA_WORKSPACE']
    if 'index' in task:
        if os.path.exists(workspace):
            logger.error(
                f'\n +------------------------------------------------------------------------------------+ \
                    \n |                                   🤖🤖🤖                                           | \
                    \n | The directory {workspace} already exists. Please remove it before indexing again.  | \
                    \n |                                   🤖🤖🤖                                           | \
                    \n +------------------------------------------------------------------------------------+'
            )
            sys.exit(1)
    if 'query' in task:
        if not os.path.exists(workspace):
            print(f'The directory {workspace} does not exist. Please index first via `python app.py -t index`')
            sys.exit(1)
    """

    if task == 'index':
        index(num_docs)
    elif task == 'query':
        query(top_k)
    elif task == 'query_restful':
        query_restful()


if __name__ == '__main__':
    os.environ['INDEX_FILENAME'] = 'data/marco_test_title_para_h100.txt.index1'
    main('query', MAX_DOCS, 5)
    # main()
