import os
import sys
import rocketqa
from rocketqa import rocketqa

de_conf = {
    "model_name": "zh_dureader_de",
    "conf_path": "/mnt/dqa/dingyuchen/irqa_jina/checkpoints/rocketqa_v2/marco/marco_de_config.json",
    "use_cuda": True,
    "gpu_card_id": 0,
    "batch_size": 16
}
ce_conf = {
    "model_name": "zh_dureader_ce",
    "conf_path": "/mnt/dqa/dingyuchen/irqa_jina/checkpoints/rocketqa_v2/marco/marco_ce_config.json",
    "use_cuda": True,
    "gpu_card_id": 0,
    "batch_size": 16
}

query_list = ["what is paula deen's brother"]
para_list = ["Paula Deen & Brother Bubba Sued for Harassment"]
title_list = ["Paula Deen and her brother Earl W. Bubba Hiers are being sued by a former general manager at Uncle Bubba'sâ<80>¦ Paula Deen and her brother Earl W. Bubba Hiers are being sued by a former general manager at Uncle Bubba'sâ"]

# init dual encoder
dual_encoder = rocketqa.load_model(de_conf)

# encode query & para
q_embs = dual_encoder.encode_query(query=query_list)
p_embs = dual_encoder.encode_para(para=para_list, title=title_list)
# compute inner product of query and para
inner_products = dual_encoder.matching(query=query_list, para=para_list, title=title_list)

# init cross encoder
cross_encoder = rocketqa.load_model(ce_conf)
# compute matching score of query and para
ranking_score = cross_encoder.matching(query=query_list, para=para_list, title=title_list)
