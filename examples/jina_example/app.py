import sys
from jina import Document, DocumentArray, Flow


def index(file_name):
    f = Flow().load_config('flows/index.yml')
    with f:
        cnt = 0
        docs = DocumentArray()
        for line in open(file_name):
            doc = Document(id=cnt, uri=file_name)
            title, para = line.strip().split('\t')
            doc.tags['title'] = title
            doc.tags['para'] = para
            cnt += 1
            docs.append(doc)
        f.post(on='/index', inputs=docs, show_progress=True)


def query():

    f = Flow().load_config('flows/query.yml')
    with f:
        #while True:
        #    text = input('please type a query: ')
        #    if not text:
        #        break
        resp = f.post(on='/search', inputs=[Document(text="what is paula deen's brother")], return_results=True)


def query_restful(return_flow=False):
    f = Flow().load_config('flows/query.yml')
    f.use_rest_gateway()
    if return_flow:
        return f
    with f:
        f.block()


def main(task):
    if task == 'index':
        index('toy_data/test.tsv')
    elif task == 'query':
        query()


if __name__ == '__main__':
    task = sys.argv[1]
    main(task)
