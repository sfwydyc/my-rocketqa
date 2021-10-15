import os
import redis
import faiss
import logging
import paddle
import numpy as np
from models.rocketqa_v1.model.src.inference_predictor import RocketPredictor
from utils.util import batch_generator,file_generator

log = logging.getLogger(__name__)
# paddle.set_device('cpu')
paddle.enable_static()

def build_id2vec_index(para_filename, encoder_conf):

    encoder = RocketPredictor(**encoder_conf)
    indexer = faiss.IndexFlatIP(768)
    with open(para_filename,'r') as f:
        for idx,batch_lines in enumerate(batch_generator(file_generator(f),encoder_conf['batch_size'])):
            if idx % 10 == 0:
                # log every 10 mini-batches
                log.info(f'{idx*encoder_conf["batch_size"]} indexed vectors')
            lines = np.char.add(np.char.add('-\t',batch_lines),'\t0')

            feats = encoder.get_cls_feats(lines)
            indexer.add(feats.astype('float32'))
    faiss.write_index(indexer,para_filename+'.index1')

def build_id2para_index(para_filename,redis_conf):
    host,port = redis_conf
    pool = redis.ConnectionPool(host=host, port=port, decode_responses=True)
    r = redis.Redis(connection_pool=pool)
    with open(para_filename,'r') as f:
        for idx,l in enumerate(f):
            if idx % 100 == 0:
                print(idx)
            r.set(idx,l.strip('\n'))


para_encoder_conf = {
    'use_cuda': True,
    'batch_size':1024,
    'init_checkpoint_dir':'models/rocketqa_v1/checkpoint/marco_dual_encoder_v2',
    'cls_type': 'para'
}
redis_conf = ('localhost',6379)
para_filename = 'data/marco_test_title_para_h100.txt'

build_id2vec_index(para_filename,para_encoder_conf)
build_id2para_index(para_filename,redis_conf)
log.info('the indexs have been build!')
