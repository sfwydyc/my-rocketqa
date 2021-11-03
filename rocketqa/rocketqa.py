import os
import sys
import paddle
import urllib
import numpy as np
import tarfile
from tqdm import tqdm
from rocketqa.predict.dual_encoder import DualEncoder
from rocketqa.predict.cross_encoder import CrossEncoder

paddle.enable_static()


__MODELS = {
        "v1_marco_de": "",
        "v1_marco_ce": "",
        "v1_nq_de": "",
        "v1_nq_ce": "",
        "pair_marco_de": "",
        "pair_nq_de": "",
        "v2_marco_de": "",
        "v2_marco_ce": "",
        "v2_nq_de": "",
        "zh_dureader_de": "",
        "zh_dureader_ce": ""
}


def available_models():
    return __MODELS.keys()


def load_model(encoder_conf):

    model_type = ''
    model_name = ''
    official_model = False
    encoder = None

    if "model_name" in encoder_conf:
        model_name = encoder_conf['model_name']
        print (model_name)
        if model_name in __MODELS:
            official_model = True
            model_path = os.path.expanduser('~/.rocketqa/') + model_name + '/'
            if not os.path.exists(model_path):
                __download(model_name)
            encoder_conf['conf_path'] = model_path + 'config.json'
            encoder_conf['model_path'] = model_path
            if model_name.find("_de") >= 0:
                model_type = 'dual_encoder'
            elif model_name.find("_ce") >= 0:
                model_type = 'cross_encoder'
        else:
            print ("not official model")

    if official_model is False:
        assert ("conf_path" in encoder_conf), "[conf_path] not found in config file"
        conf_path = encoder_conf['conf_path']
        assert (os.path.isfile(conf_path)), "[%s] not exists" %(conf_path)
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


def __download(model_name):
    os.makedirs(os.path.expanduser('~/.rocketqa/'), exist_ok=True)
    filename = model_name + '.tar.gz'
    download_dst = os.path.join(os.path.expanduser('~/.rocketqa/') + filename)
    download_url = __MODELS[model_name]
    print (download_url)

    if not os.path.exists(download_dst):
        with urllib.request.urlopen(download_url) as source, open(download_dst, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

    try:
        t = tarfile.open(download_dst)
        t.extractall(os.path.expanduser('~/.rocketqa/'))
    except Exception as e:
        print(e)



if __name__ == '__main__':
    pass
