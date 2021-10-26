import os
import sys
import paddle
import itertools
import numpy as np
from rocketqa.predict.dual_encoder import DualEncoder
from rocketqa.predict.cross_encoder import CrossEncoder

paddle.enable_static()


#def available_models():


def load_model(encoder_conf):

    model_type = ''
    model_name = ''
    encoder = None

    official_model = False
    if "model_name" in encoder_conf:
        model_name = encoder_conf['model_name']
        if model_name.find("_de") >= 0:
            model_type = 'dual_encoder'
        elif model_name.find("_ce") >= 0:
            model_type = 'cross_encoder'
        official_model = True

    if official_model is False:
        try:
            with open(conf_path, 'r', encoding='utf8') as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing model config file '%s'" %conf_path)

        assert ("model_type" in config_dict), "[model_type] not found in config file"
        model_type = config_dict["model_type"]
        assert (model_type == "dual_encoder" or model_type == "cross_encoder"), "model_type [%s] is illegal" % (m_type)

    if model_type[0] == "d":
        encoder = DualEncoder(**encoder_conf)
    elif model_type[0] == "c":
        encoder = CrossEncoder(**encoder_conf)

    return encoder

if __name__ == '__main__':
    de_conf = {
        "model_name": "v1_marco_de",
        "conf_path": "/mnt/dqa/dingyuchen/irqa_jina/checkpoints/rocketqa_v1/marco_de/marco_de_config.json",
        "use_cuda": True,
        "gpu_card_id": 2,
        "batch_size": 32
    }
    ce_conf = {
        "model_name": "zh_dureader_ce",
        "conf_path": "/mnt/dqa/dingyuchen/irqa_jina/checkpoints/rocketqa_zh/dureader_ce/dureader_ce_config.json",
        "use_cuda": True,
        "gpu_card_id": 2,
        "batch_size": 32
    }

    query_list = []
    para_list = []
    title_list = []
    marco_data = '../data/dev.qtpl.top1k'
    dureader_data = '../data/durd_test.top50.top1k'
    for line in open(dureader_data):
        q, t, p, _ = line.strip().split('\t')
        query_list.append(q)
        para_list.append(p)
        title_list.append(t)

    """
    dual_encoder = load(de_conf)
    q_embs = dual_encoder.encode_query(query=query_list)
    for q in q_embs:
        print (' '.join(str(ii) for ii in q))
    p_embs = dual_encoder.encode_para(para=para_list, title=title_list)
    for p in p_embs:
        print (' '.join(str(ii) for ii in p), file = sys.stderr)
    ips = dual_encoder.matching(query=query_list, para=para_list, title=title_list)
    #for ip in ips:
    #    print (ip)
    """

    cross_encoder = load_model(ce_conf)
    ranking_score = cross_encoder.matching(query=query_list, para=para_list, title=title_list)
    for rs in ranking_score:
        print (rs)

