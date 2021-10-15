import faiss
from jina import Document,DocumentArray, Executor, requests
import redis
import numpy as np
import sys
import logging

log = logging.getLogger(__name__)

class FaissVectorIndexer(Executor):

    def __init__(self,para_filename,top_k,**kwargs):
        super().__init__(**kwargs)
        self.engine = faiss.read_index(para_filename)
        self.topk = top_k

    @requests(on='/search-id')
    def query(self,docs: DocumentArray, **kwargs):
        embs = np.array(docs.get_attributes('embedding')) 
        res_dist,res_pid = self.engine.search(embs,self.topk)
        log.info(res_pid)
        for idx,d in enumerate(docs):
            d.matches = [Document(id=idx_) for idx_ in res_pid[idx,:]]
        log.info(d.matches)
        # return d.matches

class RedisIndexer(Executor):

    def __init__(self,host,port,**kwargs):
        super().__init__(**kwargs)
        pool = redis.ConnectionPool(
            host=host, port=port, decode_responses=True)
        self.r = redis.Redis(connection_pool=pool)

    @requests(on='/search-para')
    def query(self,docs: DocumentArray, **kwargs):
        log.info(docs[0])
        log.info(docs[0].matches)
        for d in docs:
            for match in d.matches:
                match.text = self.r.get(match.id)
                # log.info(match.text)
        # log.info(docs[0].matches.get_attributes('text'))

class FaissRedisIndexer(Executor):

    def __init__(self,index_filename,top_k,host,port,mode,**kwargs):
        super().__init__(**kwargs)
        # res = faiss.StandardGpuResources()
        # TODO: meets cuda error 3 when use faiss-gpu, fix the error later
        self.engine = faiss.read_index(index_filename)
        # self.engine = faiss.index_cpu_to_gpu(res, 0, self.engine)
        self.topk = top_k
        pool = redis.ConnectionPool(
            host=host, port=port, decode_responses=True)
        self.r = redis.Redis(connection_pool=pool)
    

    @requests
    def query(self,docs: DocumentArray, **kwargs):
        embs = np.array(docs.get_attributes('embedding'))
        res_dist,res_pid = self.engine.search(embs,self.topk)
        # the para id of marco is consistent with line number
        # so just use build_id2vec_index to build vector index
        # and when query with faiss the res_pid is the para id
        # to be evaluated
        for i,d in enumerate(docs):
            d.matches = [Document(scores={'faiss_score': res_dist[i][j]},text=p,id=res_pid[i][j]) for j,p in enumerate(self.r.mget(res_pid[i,:].tolist()))]
        # log.info(d.matches)