import os
import sys
import json
import paddle
import urllib
import numpy as np
import tarfile
import warnings
from tqdm import tqdm
from rocketqa.predict.dual_encoder import DualEncoder
from rocketqa.predict.cross_encoder import CrossEncoder

paddle.enable_static()
warnings.simplefilter('ignore')

__MODELS = {
        "v1_marco_de": "http://rocketqa.bj.bcebos.com/RocketQAModels/v1_marco_de.tar.gz",
        "v1_marco_ce": "http://rocketqa.bj.bcebos.com/RocketQAModels/v1_marco_ce.tar.gz",
        "v1_nq_de": "http://rocketqa.bj.bcebos.com/RocketQAModels/v1_nq_de.tar.gz",
        "v1_nq_ce": "http://rocketqa.bj.bcebos.com/RocketQAModels/v1_nq_ce.tar.gz",
        "pair_marco_de": "http://rocketqa.bj.bcebos.com/RocketQAModels/pair_marco_de.tar.gz",
        "pair_nq_de": "http://rocketqa.bj.bcebos.com/RocketQAModels/pair_marco_ce.tar.gz",
        "v2_marco_de": "http://rocketqa.bj.bcebos.com/RocketQAModels/v2_marco_de.tar.gz",
        "v2_marco_ce": "http://rocketqa.bj.bcebos.com/RocketQAModels/v2_marco_ce.tar.gz",
        "v2_nq_de": "http://rocketqa.bj.bcebos.com/RocketQAModels/v2_nq_de.tar.gz",
        "zh_dureader_de": "http://rocketqa.bj.bcebos.com/RocketQAModels/zh_dureader_de.tar.gz",
        "zh_dureader_ce": "http://rocketqa.bj.bcebos.com/RocketQAModels/zh_dureader_ce.tar.gz"
}


def available_models():
    """
    Return the names of available RocketQA models
    """
    return __MODELS.keys()


def load_model(model, use_cuda=False, device_id=0, batch_size=1):
    """
    Load a RocketQA model or an user-specified checkpoint
    Args:
        model: A model name return by `rocketqa.availabel_models()` or the path of an user-specified checkpoint config
        use_cuda: Whether to use GPU
        devicd_id: The device to put the model
        batch_size: Batch_size during inference
    Returns:
        model
    """

    model_type = ''
    model_name = ''
    rocketqa_model = False
    encoder_conf = {}

    if model in __MODELS:
        model_name = model
        print ("RocketQA model [{}]".format(model_name), file=sys.stderr)
        rocketqa_model = True
        model_path = os.path.expanduser('~/.rocketqa/') + model_name + '/'
        if not os.path.exists(model_path):
            if __download(model_name) is False:
                raise Exception("RocketQA model [{}] not found".format(model_name))

        encoder_conf['conf_path'] = model_path + 'config.json'
        encoder_conf['model_path'] = model_path
        if model_name.find("_de") >= 0:
            model_type = 'dual_encoder'
        elif model_name.find("_ce") >= 0:
            model_type = 'cross_encoder'

    if rocketqa_model is False:
        print ("User-specified model", file=sys.stderr)
        conf_path = model
        if not os.path.isfile(conf_path):
            raise Exception("Config file [{}] not found".format(conf_path))
        try:
            with open(conf_path, 'r', encoding='utf8') as json_file:
                config_dict = json.load(json_file)
        except Exception as e:
            raise Exception(str(e) + "\nConfig file [{}] load failed".format(conf_path))

        encoder_conf['conf_path'] = conf_path

        split_p = conf_path.rfind('/')
        if split_p > 0:
            encoder_conf['model_path'] = conf_path[0:split_p + 1]

        if "model_type" not in config_dict:
            raise Exception("[model_type] not found in config file")
        model_type = config_dict["model_type"]
        if model_type != "dual_encoder" and model_type != "cross_encoder":
            raise Exception("model_type [model_type] is illegal, must be `dual_encoder` or `cross_encoder`")

    encoder_conf["use_cuda"] = use_cuda
    encoder_conf["device_id"] = device_id
    encoder_conf["batch_size"] = batch_size

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

    if os.path.exists(download_dst):
        print ("RocketQA model [{}] exists".format(model_name), file=sys.stderr)
    else:
        print ("Download RocketQA model [{}]".format(model_name), file=sys.stderr)
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
        print (str(e), file=sys.stderr)
        return False

    return True


if __name__ == '__main__':
    pass
