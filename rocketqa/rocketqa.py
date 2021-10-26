import os
import sys
import paddle
import itertools
import numpy as np
from predict.dual_encoder import DualEncoder
from predict.cross_encoder import CrossEncoder

paddle.enable_static()


#def available_models():


def load(encoder_conf):

    model_type = ''
    model_name = ''
    find_name = False

    if "model_name" in encoder_conf:
        model_name = encoder_conf['model_name']
        if model_name.find("_de") >= 0:
            model_type = 'dual encoder'
        elif model_name.find("_ce") >= 0:
            model_type = 'cross encoder'
        find_name = True

    if find_name is False:
        try:
            with open(conf_path, 'r', encoding='utf8') as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing model config file '%s'" %conf_path)

        assert ("model_type" in config_dict), "[model_type] not found in config file"
        m_type = config_dict["model_type"]
        assert (m_type == "dual_encoder" or m_type == "cross_encoder"), "model_type [%s] is illegal" % (m_type)
        if m_type == "dual_encoder":
            model_type = "dual encoder"
        else:
            model_type = "cross encoder"

    #print ('MODEL_TYPE: ' + model_type)
    #print ('MODEL_NAME: ' + model_name)
    if model_type[0] == "d":
        return DualEncoder(**encoder_conf)
    elif model_type[0] == "c":
        return CrossEncoder(**encoder_conf)


if __name__ == '__main__':
    de_conf = {
        "model_name": "v1_marco_de",
        "conf_path": "/mnt/dqa/dingyuchen/irqa_jina/checkpoints/rocketqa_v1/marco_de/marco_de_config.json",
        "use_cuda": True,
        "gpu_card_id": 2,
        "batch_size": 32
    }
    ce_conf = {
        "model_name": "v1_marco_ce",
        "conf_path": "/mnt/dqa/dingyuchen/irqa_jina/checkpoints/rocketqa_v1/marco_ce/marco_ce_config.json",
        "use_cuda": True,
        "gpu_card_id": 2,
        "batch_size": 32
    }

    query_list = []
    para_list = []
    title_list = []
    marco_data = '../data/dev.qtpl.top1k'
    dureader_data = '../data/durd_test.top50.top1k'
    for line in open(marco_data):
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

    cross_encoder = load(ce_conf)
    ranking_score = cross_encoder.matching(query=query_list, para=para_list, title=title_list)
    for rs in ranking_score:
        print (rs)

