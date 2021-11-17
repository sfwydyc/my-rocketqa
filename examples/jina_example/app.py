import sys
from pathlib import Path
from jina import Document, DocumentArray, Flow


def index(file_name):
    f = Flow().load_config('flows/index.yml')
    with f:
        cnt = 0
        docs = DocumentArray()
        for line in open(file_name):
            doc = Document(id=f'{cnt}', uri=file_name)
            title, para = line.strip().split('\t')
            doc.tags['title'] = title
            doc.tags['para'] = para
            cnt += 1
            docs.append(doc)
        resp = f.post(on='/index', inputs=docs, show_progress=True, return_results=True)


def query():
    f = Flow().load_config('flows/query.yml')
    with f:
        resp = f.post(on='/search', inputs=[Document(text="what is paula deen's brother")], return_results=True)
    for d in resp[0].docs:
        print(f'{d.text} {d.embedding.shape}: {len(d.matches)}')
        for m in d.matches:
            print(f'+- {m.text}, {m.scores["relevance"].value}')


def main(task):
    if task == 'index':
        if Path('./workspace').exists():
            print('./workspace exists, please deleted it before reindexing')
            return
        index('../marco.tp.1k')
    elif task == 'query':
        query()


if __name__ == '__main__':
    task = sys.argv[1]
    main(task)
