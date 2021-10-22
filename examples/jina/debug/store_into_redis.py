import redis

pool = redis.ConnectionPool(host='localhost', port=6390, decode_responses=True)
r = redis.Redis(connection_pool=pool)

with open('data/dev.query.txt','r') as f:
    for idx,l in enumerate(f):
        if idx % 100 == 0:
            print(idx)
        qid,q = l.strip('\n').split('\t')
        r.hmset('pythonDict',{'qid':qid,'q':q})