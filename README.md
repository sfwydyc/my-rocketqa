# RocketQA End-to-End QA-system Development Tool

This repository provides a simple and efficient toolkit for running RocketQA models and build a QA-system. 

## RocketQA
**RocketQA** is a series of dense retrieval models for Open-Domain Question Answering. 

Open-Domain Question Answering aims to find the answers to natural language questions from a large collection of documents. Common approachs often contain two stages, firstly a dense retriever select a few revelant contexts, and then a neural reader extracts the answer.

RocketQA focus on improving the dense contexts retrieval stage, and propose the following methods:
#### 1. [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2010.08191.pdf)

#### 2. [PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval](https://aclanthology.org/2021.findings-acl.191.pdf)

#### 3. [RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking](https://arxiv.org/pdf/2110.07367.pdf)


## Features
* ***State-of-the-art***, RocketQA models achieve SOTA performance in MSMARCO passage ranking dataset and Natural Question dataset.
* ***First-Chinese-model***, RocketQA-zh is the first open source Chinese dense retrieval model.
* ***Easy-to-use***, both python installation package and DOCKER environment are provided.
* ***Solution-for-QA-system***, developers can build an End-to-End QA system with one line of code.


## Installation

### Install python package
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

NOTE: RocketQA package MUST be running on Python3.6+ with [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) 2.0+ :

### Download Docker environment

```bash
docker pull rocketqa_docker_name

docker run -it rocketqa_docker_name
```

## API
The RocketQA development tool provides two kind of models, one is ERNIE-based twin towers dual-encoder for answer retrieval, anthor is ERNIE-based cross encoder for answer reranking. And development tool provides the following methods:

#### `rocketqa.available_models()`

Returns the names of the available RocketQA models. 

#### `rocketqa.load_model(model, use_cuda=False, device_id=0, batch_size=1)`

Returns the model specifiecd by input parameter. Both dual encoder and cross encoder can be initialized by this method. With input parameter, developers can load RocketQA models returned by "available_models()" or their own checkpoints.

---

Dual-encoder returned by "load_model()" supports the following methods:

#### `model.encode_query(query: List[str])`

Given a list of queries, returns vector representations encoded by model.

#### `model.encode_para(para: List[str], )`

Given a list of passages and their titles (optional), returns the representations encoded by model.

#### `model.matching(query: List[str], para: List[str], )`

Given a list of queries and passages (titles), returns their matching scores (inner product of their representations).

---

Cross-encoder returned by "load_model()" supports the following method:

#### `model.matching(query: List[str], para: List[str], )`

Given a list of queries and passages (titles), returns their matching scoress (probability that paragraph is query's right answer).


## Examples

### Run Prediction
With the examples below, developers can run RocketQA models or their own checkpoints. 

####  Run RocketQA Model
To run RocketQA models, developers should set the parameter `model` in 'load_model()' method with RocketQA model name return by 'available_models()' method. 

```python
import rocketqa

query_list = ["trigeminal definition"]
para_list = ["Definition of TRIGEMINAL. : of or relating to the trigeminal nerve.ADVERTISEMENT. of or relating to the trigeminal nerve. ADVERTISEMENT."]

# init dual encoder
dual_encoder = rocketqa.load_model(model="v1_marco_de", use_cuda=True, batch_size=16)

# encode query & para
q_embs = dual_encoder.encode_query(query=query_list)
p_embs = dual_encoder.encode_para(para=para_list)
# compute inner product of query representation and para representation
inner_products = dual_encoder.matching(query=query_list, para=para_list, title=title_list)
```

#### Run Self-development Model
To run checkpoints, developers should write a config file, and set the parameter `model` in 'load_model()' method with the path of the config fille.

```python
import rocketqa

query_list = [""]
title_list = [""]
para_list = [""]

# conf
ce_conf = {
    "model": "./own_model/config.json",     # path of config file
    "use_cuda": True,
    "device_id": 0,
    "batch_size": 16
}

# init cross encoder
cross_encoder = rocketqa.load_model(**ce_conf)

# compute matching score of query and para
ranking_score = cross_encoder.matching(query=query_list, para=para_list, title=title_list)
```

The config file is JSON format file.
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


## Start your QA-System

With the examples blow, you can build your own QA-System

### Running with JINA




### Running with Faiss

```bash
cd examples/faiss_example/

# Index
python3 index.py ${data_file} ${index_file}

# Start service
python3 rocketqa_service.py ${data_file} ${index_file}

# request
python3 query.py
```

