import sys
import webbrowser
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
    url_html_fn = Path(__file__).parent.absolute() / 'static/index.html'
    url_html_path = f'file://{url_html_fn}'
    f = Flow().load_config('flows/query.yml')
    with f:
        try:
            webbrowser.open(url_html_path, new=2)
        except:
            pass
        finally:
            print(f'You should see a demo page opened in your browser'
                  f'if not, you may open {url_html_path} manually')
        f.block()


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
