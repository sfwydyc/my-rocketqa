import numpy as np

from jina import Document, DocumentArray, Executor, requests
from jina.types.score import NamedScore

import rocketqa


class CrossEncoder(Executor):
    def __init__(self, model, use_cuda=False, device_id=0, batch_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = rocketqa.load_model(model=model, use_cuda=use_cuda, device_id=device_id, batch_size=batch_size)
        self.b_s = batch_size

    @requests(on='/search')
    def rank(self, docs, **kwargs):
        for doc in docs:
            question = doc.text
            doc_arr = DocumentArray([doc])
            match_batches_generator = (doc_arr
                                       .traverse_flat(traversal_paths='m')
                                       .batch(batch_size=self.b_s))

            reranked_matches = DocumentArray()
            reranked_scores = []
            unsorted_matches = DocumentArray()
            for matches in match_batches_generator:
                titles, paras = matches.get_attributes('tags__title', 'tags__para')
                score_list = self.encoder.matching(query=[question] * len(paras), para=paras, title=titles)
                reranked_scores.extend(score_list)
                unsorted_matches += list(matches)
            sorted_args = np.argsort(reranked_scores).tolist()
            sorted_args.reverse()
            for idx in sorted_args:
                score = reranked_scores[idx]
                m = Document(
                    id=unsorted_matches[idx].id,
                    tags={
                        'title': unsorted_matches[idx].tags['title'],
                        'para': unsorted_matches[idx].tags['para']
                    }
                )
                m.scores['relevance'] = NamedScore(value=score)
                reranked_matches.append(m)
            doc.matches = reranked_matches
