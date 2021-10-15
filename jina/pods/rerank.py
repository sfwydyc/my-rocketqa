from models.rocketqa_v1.model.src.rocket_rerank import RocketReRanker
from jina import DocumentArray, Executor, Flow, requests
import numpy as np

class RocketQAReRanker(Executor):

    def __init__(self,use_cuda,batch_size,init_checkpoint_dir,**kwargs):
        super().__init__(**kwargs)
        self.model = RocketReRanker(use_cuda,batch_size,init_checkpoint_dir)

    @requests(on='/rerank')
    def rerank(self,docs: DocumentArray, **kwargs):
        for d in docs:
            data = np.char.add(np.char.add(d.text+'\t', d.matches.get_attributes('text')),'\t0')
            #print(data)
            #print(data[0])
            new_scores = self.model.get_scores(data)
            for match,new_score in zip(d.matches,new_scores):
                match.scores = {'rerank_score':new_score}
                #print(match.text)
                #print(match.scores['rerank_score'].value)
            d.matches.sort(key=lambda x: x.scores['rerank_score'].value,reverse=True)
