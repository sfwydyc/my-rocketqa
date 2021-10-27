# RocketQA
RocketQA is a simple and efficient toolkit for running Open-Domain Question Answering models. It provides dense passage retrieval and ranking models for both English and Chinese. The English models can reach SOTA performance in MSMARCO and Natural Question datasset.

## Features
* Convenient interface, developers can index dataset and search with a few lines of Python code.
* Various models are supported, including RocketQA-v1, PAIR, RocketQA-v2 and RocketQA-zh, which is the first open source Chinese dense retrieval model. 
* 

## Approach

### RocketQA-v1: [[paper]](https://arxiv.org/pdf/2010.08191.pdf)  [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/NAACL2021-RocketQA)

### PAIR: [[paper]](https://aclanthology.org/2021.findings-acl.191.pdf)  [[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2021-PAIR)

### RocketQA-v2: [[paper]](https://arxiv.org/pdf/2110.07367.pdf) [[code]]()


## Installation

First, install [PaddlePaddle 2.1+](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) as well as additional dependencies:
```bash
$ pip install paddlepaddle-gpu
```
Then, install RocketQA package:
```bash
$ pip install rocketqa
```

## API
The RocketQA module rocketqa provides the following methods:

#### `rocketqa.available_models()`

Returns the names of the available models

#### `rocketqa.load_model(encoder_config)`

Returns the model specifiecd by input encoder_config. 
With encoder config, developers can choose the RocketQA models returned by available_models(), or their own training checkpoints.

---

The model returned by load_model() supports the following methods:

#### `model.encoder_query(query: string list)`

Given a list of queries, returns the representations encoded by model.

#### `model.encoder_para(para: para list, ...)`

Given a list of passages and their titles (optional), returns the representations encoded by model.

#### `model.matching(query: string list, para: para list)`

Given a list of queries and passages (titles), returns their matching scores.


## Examples



