# RocketQA
RocketQA is a series of dense retrieval models for Open-Domain Question Answering. 

Open-Domain Question Answering aims to find the answers to natural language questions from a large collection of documents. Common approachs often contain two stages, firstly a dense retriever select a few revelant contexts, and then a neural reader extracts the answer.

RocketQA focus on improving the dense contexts retrieval stage, and propose the following methods:
#### 1. [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2010.08191.pdf)  [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/NAACL2021-RocketQA)

#### 2. [PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval](https://aclanthology.org/2021.findings-acl.191.pdf)  [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2021-PAIR)

#### 3. [RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking](https://arxiv.org/pdf/2110.07367.pdf) [[code]]()

This repository provides a simple and efficient toolkit for running RocketQA models and build a QA-system. 

This toolkit is currently under initial development stage. We will be actively adding new features and models. Suggestions, feature requests

## Features
* ***State-of-the-art***, RocketQA models achieve SOTA performance in MSMARCO passage ranking dataset and Natural Question dataset.
* ***First-Chinese-model***, RocketQA-zh is the first open source Chinese dense retrieval model.
* ***Easy-to-use***, require two lines of Python code to get query/passage's representation.
* ***Complete-solution-for-QA***, dense retriever, passage ranker and neural reader is provided, require a few lines of code to build a QA system.


## Installation

First, install [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) as well as additional dependencies:
```bash
$ pip install paddlepaddle-gpu
```
Then, install RocketQA package:
```bash
$ pip install rocketqa
```
NOTE: RocketQA package MUST be running on Python3.6+ with PaddlePaddle 2.1+.

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



