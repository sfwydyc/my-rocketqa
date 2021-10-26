# RocketQA
RocketQA is a simple and efficient toolkit for running dense passage retrievers with deep language models. It provides models for both English and Chinese, and the English models can reach SOTA performance in MSMARCO and Natural Question datasset.

## Features
* Convenient interface for newcomers
* Various models are supported, including RocketQA-v1, PAIR, RocketQA-v2 and RocketQA-zh, which is the first open source Chinese dense retrieval model. 

## Installation
First, install PaddlePaddle 2.0+, as well as additional dependencies:

Then, install this package:

## API
The RocketQA module rocketqa provides the following methods:

rocketqa.available_models()

Returns the names of the available models

rocketqa.load_model(encoder_config)

Returns the model specifiecd by input encoder_config. 
With encoder config, developers can choose the RocketQA series models in available_models() or his own checkpoints.


The model returned by load_model() supports the following methods:

model.encoder_query(query: string list)

Given a list of queries, returns the representations encoded by model.

model.encoder_para(para: para list, ...)

Given a list of passages and their titles (optional), returns the representations encoded by model.

model.matching(query: string list, para: para list)

Given a list of queries and a list of passages, returns their matching scores.

## Examples


## Approach

### RocketQA-v1

### PAIR

### RocketQA-v2


