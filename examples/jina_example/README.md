# RocketQA❤️Jina

This is a simple Demo of how to use RocketQA together with Jina.

## Usages

### Install Dependencies
```shell
cd examples/jina_example
pip install -r requirements.txt
```

### Index

Run the following line to index the data stored at `toy_data/test.tsv`

```shell
python app.py index toy_data/test.tsv
```

The data is formatted as below
```text
title_1\ttext_1\n
title_2\ttext_2\n
...
```

### Query

Run the following line to start the query service. You will have a demo page opened in your browser. By default, port `8889` is used. 

```shell
python app.py query
```

If you prefer to use the shell, please try

```shell
python app.py query_cli
```