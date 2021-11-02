import os
import faiss
import logging
import rocketqa
from rocketqa import rocketqa

log = logging.getLogger(__name__)


def build_index(encoder_conf, index_file_name, title_list, para_list):

    dual_encoder = rocketqa.load_model(encoder_conf)
    para_embs = dual_encoder.encode_para(para=para_list, title=title_list)

    indexer = faiss.IndexFlatIP(768)
    emb_f = open('marco_paraemb_all.bin', 'wb')
    for emb in para_embs:
        emb_str = ' '.join(str(score) for score in emb) + '\n'
        emb_f.write(emb_str.encode(encoding='utf8'))
    indexer.add(para_embs.astype('float32'))
    faiss.write_index(indexer, index_file_name)


if __name__ == '__main__':
    de_conf = {
        "model_name": "v1_marco_de",
        "conf_path": "",
        "use_cuda": True,
        "gpu_card_id": 0,
        "batch_size": 32
    }

    marco_data = 'marco.tp.1k'
    dureader_data = 'durd_test.top50.top1k'
    para_list = []
    title_list = []
    for line in open(marco_data):
        t, p = line.strip().split('\t')
        para_list.append(p)
        title_list.append(t)

    build_index(de_conf, 'marco_test.index', title_list, para_list)
