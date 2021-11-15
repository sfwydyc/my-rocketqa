import numpy as np

from jina import Document, DocumentArray, Executor, requests
from jina.types.score import NamedScore
from jina.logging.logger import JinaLogger

import rocketqa


class DualEncoder(Executor):
    def __init__(self, model, use_cuda=False, device_id=0, batch_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(str(self.__class__))
        self.encoder = rocketqa.load_model(model=model, use_cuda=use_cuda, device_id=device_id, batch_size=batch_size)
        self.b_s = batch_size

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
                doc.embedding = emb.squeeze()

    @requests(on='/search')
    def encode_search(self, docs, **kwargs):
        for doc in docs:
            query_emb = self.encoder.encode_query(query=[doc.text])
            doc.embedding = query_emb.squeeze()


class CrossEncoder(Executor):
    def __init__(self, model, use_cuda=False, device_id=0, batch_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(str(self.__class__))
        self.encoder = rocketqa.load_model(model=model, use_cuda=use_cuda, device_id=device_id, batch_size=batch_size)
        self.b_s = batch_size

    @requests(on='/search')
    def rank(self, docs, **kwargs):
        for doc in docs:
            question = doc.text
            doc_arr = DocumentArray([doc])
            match_batches_generator = (doc_arr
                                       .traverse_flat(traversal_paths=('m',))
                                       .batch(batch_size=self.b_s))

            reranked_matches = DocumentArray()
            reranked_scores = []
            reranked_texts = []
            for matches in match_batches_generator:
                titles, paras = matches.get_attributes('tags__title', 'tags__para')
                q_list = []
                while len(q_list) < len(paras):
                    q_list.append(question)
                score_list = self.encoder.matching(query=q_list, para=paras, title=titles)
                reranked_scores.extend(score_list)
                reranked_texts.extend([f'{title}\t{para}' for title, para in zip(titles, paras)])
            sorted_args = np.argsort(reranked_scores).tolist()
            sorted_args.reverse()
            for idx in sorted_args:
                score = reranked_scores[idx]
                text = reranked_texts[idx]
                m = Document(text=f'{text}')
                m.scores['relevance'] = NamedScore(value=score)
                reranked_matches.append(m)
            doc.matches = reranked_matches
