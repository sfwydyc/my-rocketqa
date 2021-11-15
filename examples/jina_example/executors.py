import numpy as np

from jina import Document, DocumentArray, Executor, requests
from jina.types.score import NamedScore
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
    def encode_index(self, docs, **kwargs):
        batch_generator = (docs
                           .traverse_flat(
                                traversal_paths=('r',),
                                filter_fn=lambda d: d.tags.get('title', None) is not None and d.tags.get('para', None) is not None)
                           .batch(batch_size=32))
        for batch in batch_generator:
            titles, paras = batch.get_attributes('tags__title', 'tags__para')
            para_embs = self.encoder.encode_para(para=paras, title=titles)
            for doc, emb in zip(batch, para_embs):
                doc.embedding = emb

    @requests(on='/search')
    def encode_search(self, docs, **kwargs):
        for doc in docs:
            query_emb = self.encoder.encode_query(query=[doc.text])
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
            question = doc.text
            doc_arr = DocumentArray([doc])
            match_batches_generator = (doc_arr
                .traverse_flat(
                    traversal_paths=('m', ),
                    filter_fn=lambda d: d.text)
                .batch(
                    batch_size=self.b_s))
            # self.logger.info(doc.text)
            # self.logger.info('ANN Ranker !!!')
            # self.logger.info('ann matches : ' + str(len(doc.matches)))

            reranked_matches = DocumentArray()
            for matches in match_batches_generator:
                # logger.info(matches[0])
                # question = doc_arr[0].text
                titles, paras = matches.get_attributes('tags__title', 'tags__paras')
                score_list = self.encoder.matching(query=self.fill_in(question), para=paras, title=titles)
                sorted_args = np.argsort(score_list).tolist()
                sorted_args.reverse()
                for idx in sorted_args:
                    para = paras[idx]
                    title = titles[idx]
                    m = Document(text=f'{title}\t{para}')
                    m.scores['relevance'] = NamedScore(value=score_list[idx])
                    reranked_matches.append(m)
            doc.matches = reranked_matches
