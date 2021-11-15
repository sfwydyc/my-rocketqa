import os
import click

os.environ['FLAGS_eager_delete_tensor_gb'] = '0'
paddle.enable_static()

def print_topk(req_obj, sentence, rerank=False):
    for d in req_obj.data.docs:
        for idx, match in enumerate(d.matches):
            if rerank:
                score = match.scores['rerank_score'].value
            else:
                score = match.scores['faiss_score'].value
            print(f'> {idx:>2d}({score:.2f}). {match.text}')
            #print("++++++++" + match.text.split('\t')[0] + "++++++++++")
            # print(match.text)

# @click.command()
# @click.option(
#     '--rerank',
#     '-r',
#     type=click.Choice(['True', 'False']),
# )
# @click.option('--top_k', '-k', default=5)
def main(top_k, index_filename, rerank=False):
    os.environ['QUERY_TOPK'] = os.environ.get('QUERY_TOPK', str(top_k))
    os.environ['INDEX_FILENAME'] = os.environ.get('INDEX_FILENAME', index_filename)
    #os.environ['JINA_MP_START_METHOD'] = 'spawn'

    f = Flow().add(uses='jina/pods/query-encode.yml').add(uses='jina/pods/query-index.yml').add(uses='jina/pods/res-rank.yml')
    with f:
        while True:
            text = input('please type a query: ')
            if not text:
                break
            if rerank is True:
                f.post('/rerank',Document(text=text),on_done=lambda x: print_topk(x,text, True))
            else:
                f.post('/',Document(text=text),on_done=lambda x: print_topk(x,text, False))

if __name__ == '__main__':
    main(5,'data/marco_test_title_para_h100.txt.index1', False)
    #main(6,'data/title_para_all.txt.index',True)
