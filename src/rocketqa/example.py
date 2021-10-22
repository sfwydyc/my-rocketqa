import os
import paddle
import itertools
import numpy as np
from predict.dual_encoder import DualEncoder
from predict.cross_encoder import CrossEncoder

paddle.enable_static()

def EncoderExample(encoder_conf, data_list, side):
    encoder = DualEncoder(**encoder_conf)
    feats = encoder.get_representation(data_list)
    emb_f = open(side + '_emb', 'wb')
    for fff in feats:
        emb_str = ' '.join(str(score) for score in fff) + '\n'
        emb_f.write(emb_str.encode(encoding='utf8'))


def RankerExample(encoder_conf, data_list):
    encoder = CrossEncoder(**encoder_conf)
    scores = encoder.get_rank_score(data_list)
    score_f = open('score', 'w')
    for s in scores:
        score_f.write(str(s) + '\n')


if __name__ == '__main__':
    para_encoder_conf = {
        "conf_path": "/mnt/dqa/dingyuchen/irqa_jina/checkpoints/rocketqa_v1/marco_de_config.json",
        "use_cuda": True,
        "gpu_card_id": 2,
        "batch_size": 32,
        "cls_type": "para"
    }

    query_encoder_conf = {
        "conf_path": "/mnt/dqa/dingyuchen/irqa_jina/checkpoints/rocketqa_v1/marco_de_config.json",
        "use_cuda": True,
        "gpu_card_id": 2,
        "batch_size": 32,
        "cls_type": "query"
    }

    rank_conf = {
        "conf_path": "/mnt/dqa/dingyuchen/irqa_jina/checkpoints/rocketqa_v1/marco_ce_config.json",
        "use_cuda": True,
        "gpu_card_id": 2,
        "batch_size": 32
    }


    data_list = []
    for line in open("../../data/dev.qtpl.top50"):
        v = line.strip().split('\t')
        data_list.append('\t'.join(v[:3]))

    #EncoderExample(para_encoder_conf, data_list, 'para')
    #EncoderExample(query_encoder_conf, data_list, 'query')
    RankerExample(rank_conf, data_list)
