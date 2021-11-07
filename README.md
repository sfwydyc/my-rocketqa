# RocketQA Dense Retrieval Tool

This repository provides a simple and efficient toolkit for running RocketQA models and build a QA-system. 

## RocketQA
**RocketQA** is a series of dense retrieval models for Open-Domain Question Answering. 

Open-Domain Question Answering aims to find the answers to natural language questions from a large collection of documents. Common approachs often contain two stages, firstly a dense retriever select a few revelant contexts, and then a neural reader extracts the answer.

RocketQA focus on improving the dense contexts retrieval stage, and propose the following methods:
#### 1. [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2010.08191.pdf)  [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/NAACL2021-RocketQA)

#### 2. [PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval](https://aclanthology.org/2021.findings-acl.191.pdf)  [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2021-PAIR)

#### 3. [RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking](https://arxiv.org/pdf/2110.07367.pdf) [[code]]()


## Features
* ***State-of-the-art***, RocketQA models achieve SOTA performance in MSMARCO passage ranking dataset and Natural Question dataset.
* ***First-Chinese-model***, RocketQA-zh is the first open source Chinese dense retrieval model.
* ***Easy-to-use***, require two lines of Python code to get query/passage's representation.
* ***Solution-for-QA-system***, dense retriever, passage ranker and neural reader is provided, allowing developers to build a QA system with a few lines of code.


## Installation

```bash
$ pip install rocketqa
```
NOTE: RocketQA package MUST be running on Python3.6+ with [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) 2.0+ :

## API
The RocketQA module rocketqa provides the following methods:

#### `rocketqa.available_models()`

Returns the names of the available models

#### `rocketqa.load_model(encoder_config)`

Returns the model specifiecd by input encoder_config. 
With encoder config, developers can choose the RocketQA models returned by available_models(), or their own training checkpoints.

---

The model returned by load_model() supports the following methods:

#### `model.encoder_query(query: List[str])`

Given a list of queries, returns the representations encoded by model.

#### `model.encoder_para(para: List[str], )`

Given a list of passages and their titles (optional), returns the representations encoded by model.

#### `model.matching(query: List[str], para: List[str], )`

Given a list of queries and passages (titles), returns their matching scores.


## Examples

### Run Prediction
With the code below, you can run RocketQA models or your own checkpoints. To run RocketQA models, xxx . To run your own checkpoints, xxxxxx .

####  Run RocketQA Model

```python
import rocketqa

de_conf = {
    "model": "zh_dureader_de",  
    "use_cuda": True,
    "device_id": 0,
    "batch_size": 16
}

query_list = ["what is paula deen's brother"]
para_list = ["Paula Deen & Brother Bubba Sued for Harassment"]
title_list = ["Paula Deen and her brother Earl W. Bubba Hiers are being sued by a former general manager at Uncle Bubba'sâ<80>¦ Paula Deen and her brother Earl W. Bubba Hiers are being sued by a former general manager at Uncle Bubba'sâ"]

# init dual encoder
dual_encoder = rocketqa.load_model(**de_conf)

# encode query & para
q_embs = dual_encoder.encode_query(query=query_list)
p_embs = dual_encoder.encode_para(para=para_list, title=title_list)
# compute inner product of query and para
inner_products = dual_encoder.matching(query=query_list, para=para_list, title=title_list)
```


#### Run Self-development Model

```python
import rocketqa

# conf
ce_conf = {
    "model": "path of your own config file",
    "use_cuda": True,
    "device_id": 0,
    "batch_size": 16
}

query_list = ["what is paula deen's brother"]
para_list = ["Paula Deen & Brother Bubba Sued for Harassment"]
title_list = ["Paula Deen and her brother Earl W. Bubba Hiers are being sued by a former general manager at Uncle Bubba'sâ<80>¦ Paula Deen and her brother Earl W. Bubba Hiers are being sued by a former general manager at Uncle Bubba'sâ"]
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
    "model_conf_path": "en_large_config.json",
    "model_vocab_path": "en_vocab.txt",
    "model_checkpoint_path": "marco_cross_encoder_large",
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



