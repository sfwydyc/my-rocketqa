import sys
import requests
import json

SERVICE_ADD = 'http://10.12.35.41:8888/rocketqa'
TOPK = 5
QUERY_FILE = '../marco.q'

for line in open(QUERY_FILE, 'r'):
    query = line.strip()

    input_data = {}
    input_data['query'] = query
    input_data['topk'] = TOPK
    json_str = json.dumps(input_data)

    result = requests.post(SERVICE_ADD, json=input_data)
    res_json = json.loads(result.text)

    print ("QUERY:\t" + query)
    for i in range(TOPK):
        title = res_json['answer'][i]['title']
        para = res_json['answer'][i]['para']
        score = res_json['answer'][i]['probability']
        print ('{}'.format(i + 1) + '\t' + title + '\t' + para + '\t' + str(score))
    break



