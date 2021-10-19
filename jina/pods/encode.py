from models.rocketqa_v1.model.src.predict_de import DEPredictor
from jina import DocumentArray, Executor, requests
import numpy as np

class RocketQAEncoder(Executor):

    def __init__(self, conf_path, use_cuda, gpu_card_id, batch_size, cls_type, **kwargs):
        super().__init__(**kwargs)
        self.model = DEPredictor(conf_path, use_cuda, gpu_card_id, batch_size, cls_type)

    @requests
    def encode(self, docs: DocumentArray, **kwargs):
        content = docs.get_attributes('text')
        debug_f = open('dyc_debug', 'w')
        if self.model.cls_type == 'para':
            content = np.char.add('-\t',content)
        elif self.model.cls_type == 'query':
            content = np.char.add(content,'\t-\t-')
        else:
            assert False, 'the cls type is wrong!'
        embs = self.model.get_cls_feats(content)
        for doc, emb in zip(docs, embs):
            doc.embedding = emb
            debug_f.write(' '.join(str(score) for score in emb) + '\n')
