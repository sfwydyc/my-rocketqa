from models.rocketqa_v1.model.src.inference_predictor import RocketPredictor
from jina import DocumentArray, Executor, requests
import numpy as np

class RocketQAEncoder(Executor):

    def __init__(self,use_cuda,batch_size,init_checkpoint_dir,cls_type,**kwargs):
        super().__init__(**kwargs)
        self.model = RocketPredictor(use_cuda,batch_size,init_checkpoint_dir,cls_type)

    @requests
    def encode(self, docs: DocumentArray, **kwargs):
        content = docs.get_attributes('text')
        if self.model.cls_type == 'para':
            content = np.char.add(np.char.add('-\t',content),'\t0')
        elif self.model.cls_type == 'query':
            content = np.char.add(content,'\t-\t-\t0')
        else:
            assert False, 'the cls type is wrong!'
        embs = self.model.get_cls_feats(content)
        for doc,emb in zip(docs,embs):
            doc.embedding = emb
