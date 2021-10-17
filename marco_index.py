import os
import redis
import faiss
import logging
import paddle
import itertools
import numpy as np
from models.rocketqa_v1.model.src.predict_de import DEPredictor


batch_size = 32
log = logging.getLogger(__name__)
# paddle.set_device('cpu')
paddle.enable_static()

def file_generator(iterable):
    for line in iterable:
        yield '-\t' + line.strip()

def batch_generator(iterable, batch_size=1):
    iterable = iter(iterable)
    while True:
        batch = list(itertools.islice(iterable, batch_size))
        if batch:
            yield batch
        else:
            break

def build_id2vec_index(para_filename, encoder_conf):

    encoder = DEPredictor(**encoder_conf)
    indexer = faiss.IndexFlatIP(768)
    with open(para_filename, 'r') as f:
        for idx, batch_lines in enumerate(batch_generator(file_generator(f), batch_size)):
            if idx % 100 == 0:
                # log every 10 mini-batches
                log.info(f'{idx*batch_size} indexed vectors')

            feats = encoder.get_cls_feats(batch_lines)
            for fff in feats:
                print (' '.join(str(score) for score in fff))
            indexer.add(feats.astype('float32'))
    faiss.write_index(indexer,para_filename+'.index1')

def build_id2para_index(para_filename, redis_conf):
    host,port = redis_conf
    pool = redis.ConnectionPool(host=host, port=port, decode_responses=True)
    r = redis.Redis(connection_pool=pool)
    with open(para_filename,'r') as f:
        for idx, l in enumerate(f):
            if idx % 100 == 0:
                print(idx)
            r.set(idx,l.strip('\n'))


para_encoder_conf = {
    "conf_path": "models/rocketqa_v1/model/config/de_config.json",
    "use_cuda": True,
    "gpu_card_id": 5,
    "cls_type": "para"
}
redis_conf = ('localhost',6379)
para_filename = "data/marco_test_title_para_h100.txt"

build_id2vec_index(para_filename, para_encoder_conf)
build_id2para_index(para_filename, redis_conf)
log.info('the indexs have been build!')
