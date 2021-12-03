# RocketQA

In recent years, the dense retrievers based on pre-trained language models have achieved remarkable progress. To facilitate more developers to easily use cutting edge technologies, this repository provides an easy-to-use toolkit for running and fine-tuning the state-of-the-art dense retrievers, namely **RocketQA**. This toolkit has the following advantages:


* ***State-of-the-art***: This toolkit provides well-trained RocketQA models, which achieve SOTA performance on many dense retrieval dataset, and will continue to update the latest models.
* ***First-Chinese-model***: This toolkit provides the first open source Chinese dense retrieval model, which is trained on millions of manual annotation query-passage data from [DuReader dataset](https://github.com/baidu/DuReader).
* ***Easy-to-use***: By integrating with [JINA](https://jina.ai/), this toolkit provides an example to build an end-to-end question answering system with several lines of code.

  

## Installation

We provide two installation methods: ***Python Installation Package*** and ***DOCKER Environment***

### Install with Python Package
First, install [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html).
```bash
# GPU version:
$ pip install paddlepaddle-gpu

# CPU version:
$ pip install paddlepaddle
```

Second, install rocketqa package:
```bash
$ pip install rocketqa
```

NOTE: this toolkit MUST be running on Python3.6+ with [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) 2.0+ :

### Install with Docker

```bash
docker pull rocketqa/rocketqa

docker run -it docker.io/rocketqa/rocketqa bash
```

## Start your Question Answer (QA) System

With the examples below, you can build your own QA-System.

### Running with JINA
[JINA](https://jina.ai/) is A cloud-native neural search framework to build SOTA and scalable deep learning search applications in minutes.
Please view [JINA example](https://github.com/PaddlePaddle/RocketQA/tree/main/examples/jina_example) to see our example.


### Running with Faiss
We also provide an example built on [Faiss](https://github.com/facebookresearch/faiss)
```bash
cd examples/faiss_example/
pip3 install -r requirements.txt

# Index
python3 index.py ${language} ${data_file} ${index_file}

# Start service
python3 rocketqa_service.py ${language} ${data_file} ${index_file}

# request
python3 query.py
```


## API
RocketQA provides two types of models, ERNIE-based dual encoder for answer retrieval and ERNIE-based cross encoder for answer re-ranking. To run RocketQA models and build your own QA-system, you can use the following functions.

### Load model

#### [`rocketqa.available_models()`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/rocketqa.py#L17)

Returns the names of the available RocketQA models.
To know more about the available models, please click the function namei and see the code comment.

#### [`rocketqa.load_model(model, use_cuda=False, device_id=0, batch_size=1)`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/rocketqa.py#L52)

Returns the model specified by the input parameter. Both dual encoder and cross encoder can be initialized by this method. With input parameter, you can load RocketQA models returned by "available_models()" or their own checkpoints.

### Dual encoder
Dual-encoder returned by "load_model()" supports the following methods:

#### [`model.encode_query(query: List[str])`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/predict/dual_encoder.py#L126)

Given a list of queries, returns their representation vectors encoded by model.

#### [`model.encode_para(para: List[str], title: List[str])`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/predict/dual_encoder.py#L154)

Given a list of passages and their corresponding titles (optional), returns their representations vectors encoded by model.

#### [`model.matching(query: List[str], para: List[str], title: List[str])`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/predict/dual_encoder.py#L187)

Given a list of queries and passages (and titles), returns their matching scores (dot product between two representation vectors). 

### Croess encoder
Cross-encoder returned by "load_model()" supports the following method:

#### [`model.matching(query: List[str], para: List[str], title: List[str])`](https://github.com/PaddlePaddle/RocketQA/blob/3a99cf2720486df8cc54acc0e9ce4cbcee993413/rocketqa/predict/cross_encoder.py#L129)

Given a list of queries and passages (and titles), returns their matching scores (probability that the paragraph is the query's right answer).
  
  

## Examples

Following the examples below, you can run RocketQA models and your own checkpoints. 

###  Run RocketQA Model
To run RocketQA models, you should set the parameter `model` in 'load_model()' method with RocketQA model name return by 'available_models()' method.

```python
import rocketqa

query_list = ["trigeminal definition"]
para_list = [
    "Definition of TRIGEMINAL. : of or relating to the trigeminal nerve.ADVERTISEMENT. of or relating to the trigeminal nerve. ADVERTISEMENT."]

# init dual encoder
dual_encoder = rocketqa.load_model(model="v1_marco_de", use_cuda=True, batch_size=16)

# encode query & para
q_embs = dual_encoder.encode_query(query=query_list)
p_embs = dual_encoder.encode_para(para=para_list)
# compute dot product of query representation and para representation
dot_products = dual_encoder.matching(query=query_list, para=para_list)
```

### Run Self-development Model
To run your own checkpoints, you should write a config file, and set the parameter `model` in 'load_model()' method with the path of the config file.

```python
import rocketqa

query_list = ["交叉验证的作用"]
title_list = ["交叉验证的介绍"]
para_list = ["交叉验证(Cross-validation)主要用于建模应用中，例如PCR 、PLS回归建模中。在给定的建模样本中，拿出大部分样本进行建模型，留小部分样本用刚建立的模型进行预报，并求这小部分样本的预报误差，记录它们的平方加和。"]

# conf
ce_conf = {
    "model": ${YOUR_CONFIG},     # path of config file
    "use_cuda": True,
    "device_id": 0,
    "batch_size": 16
}

# init cross encoder
cross_encoder = rocketqa.load_model(**ce_conf)

# compute matching score of query and para
ranking_score = cross_encoder.matching(query=query_list, para=para_list, title=title_list)
```

${YOUR_CONFIG} is a JSON format file.
```bash
{
    "model_type": "cross_encoder",
    "max_seq_len": 160,
    "model_conf_path": "en_large_config.json",  # path relative to config file
    "model_vocab_path": "en_vocab.txt",         # path relative to config file
    "model_checkpoint_path": "marco_cross_encoder_large", # path relative to config file
    "joint_training": 0
}
```
  


#### 1. [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2010.08191.pdf)

#### 2. [PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval](https://aclanthology.org/2021.findings-acl.191.pdf)

#### 3. [RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking](https://arxiv.org/pdf/2110.07367.pdf)

## News


## License
PaddlePaddle is provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/RocketQA/blob/main/LICENSE).
