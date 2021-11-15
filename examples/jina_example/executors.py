import sys
import requests as py_requests

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from jina import Document, DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger

import rocketqa

class DualEncoder(Executor):
    def __init__(self, model, use_cuda=False, device_id=0, batch_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger('RocketQA-Executor')
        self.encoder = rocketqa.load_model(model=model, use_cuda=use_cuda, device_id=device_id, batch_size=batch_size)
        self.b_s = batch_size
        self.logger.info('Retriever init done')

    @requests(on='/index')
    def encode(self, docs, **kwargs):
        fff = open('p_emb', 'a')
        for batch in docs.batch(batch_size=32):
            tags = batch.get_attributes('tags')
            titles = []
            paras = []
            for tag in tags:
                titles.append(tag['title'])
                paras.append(tag['para'])
            #for i in  range(len(titles)):
            #    fff.write(titles[i] + '\t' + paras[i] + '\n')
            para_embs = self.encoder.encode_para(para=paras, title=titles)
            for emb in para_embs:
                fff.write(' '.join(str(ii) for ii in emb) + '\n')
            batch.embedding = para_embs

    @requests(on='/search')
    def encode(self, docs, **kwargs):
        fff = open('q_emb', 'w')
        for doc in docs:
            query = doc.text
            #fff.write(query + '\n')
            query_emb = self.encoder.encode_query(query=[query])
            fff.write(' '.join(str(ii) for ii in query_emb[0]) + '\n')
            doc.embedding = query_emb


class CrossEncoder(Executor):
    def __init__(self, model, use_cuda=False, device_id=0, batch_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(str(self.__class__))
        self.encoder = rocketqa.load_model(model=model, use_cuda=use_cuda, device_id=device_id, batch_size=batch_size)
        self.b_s = batch_size
        self.logger.info('Reranker init done')

    @requests(on='/search')
    def rank(self, docs, **kwargs):
        if not docs:
            return None

        for doc in docs:
            doc_arr = DocumentArray([doc])
            match_batches_generator = doc_arr.batch(
                traversal_paths=['m'],
                batch_size=self.b_s,
                require_attr='text',
            )
            self.logger.info(doc.text)
            self.logger.info('ANN Ranker !!!')
            self.logger.info('ann matches : ' + str(len(doc.matches)))

            for matches in match_batches_generator:
                # logger.info(matches[0])
                question = doc_arr[0].text
                oris = matches.get_attributes('text')
                titles = matches.get_attributes('tag')

