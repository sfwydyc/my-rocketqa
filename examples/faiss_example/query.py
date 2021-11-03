import sys
import requests
import json

SERVICE_ADD = 'http://localhost:8888/rocketqa'
TOPK = 10
QUERY_FILE = 'marco.q'

for line in open(QUERY_FILE, 'r'):
    query = line.strip()

    input_data = {}
    input_data['query'] = query
    input_data['topk'] = TOPK
    json_str = json.dumps(input_data)

    result = requests.post(SERVICE_ADD, json_str)
    res_json = json.loads(result.text)

    for i in range(TOPK):
        title = res_json['answer'][i]['title']
        para = res_json['answer'][i]['para']
        score = res_json['answer'][i]['probability']
        print (query + '\t' + title + '\t' + para + '\t' + str(score))
        break

