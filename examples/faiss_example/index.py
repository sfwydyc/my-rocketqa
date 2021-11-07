import os
import sys
import faiss
import rocketqa


def build_index(encoder_conf, index_file_name, title_list, para_list):

    dual_encoder = rocketqa.load_model(**encoder_conf)
    para_embs = dual_encoder.encode_para(para=para_list, title=title_list)

    indexer = faiss.IndexFlatIP(768)
    emb_f = open('marco_paraemb_all.bin', 'wb')
    for emb in para_embs:
        emb_str = ' '.join(str(score) for score in emb) + '\n'
        emb_f.write(emb_str.encode(encoding='utf8'))
    indexer.add(para_embs.astype('float32'))
    faiss.write_index(indexer, index_file_name)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print ("USAGE: ")
        print ("      python3 index.py ${data_file} ${index_file}")
        print ("    --For Example:")
        print ("      python3 index.py ../marco.tp.1k marco_test.index")
        exit()

    data_file = sys.argv[1]
    index_file = sys.argv[2]
    para_list = []
    title_list = []
    for line in open(data_file):
        t, p = line.strip().split('\t')
        para_list.append(p)
        title_list.append(t)

    de_conf = {
            "model":"v1_marco_de",
            "use_cuda":True,
            "device_id":0,
            "batch_size": 32
    }
    build_index(de_conf, index_file, title_list, para_list)
