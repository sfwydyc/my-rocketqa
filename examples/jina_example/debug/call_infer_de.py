from collections import namedtuple
from rocket_qa.model.src import inference_de
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # set device
import paddle
paddle.enable_static()

args = {
    'use_cuda': True,
    'use_fast_executor': True,
    'do_train': False,
    'do_val': False,
    'do_test': False,
    'batch_size': 1, # 256
    'init_checkpoint': 'rocket_qa/checkpoint/marco_dual_encoder_v2',
    # 'test_set': 'rocket_qa/corpus/marco/dev.query.txt.format',
    'test_set': 'rocket_qa/corpus/marco/para_8part/part-00_500',
    'test_save': 'output/test_out.tsv',
    # 'output_item': 0,
    'output_item': 1,
    # 'output_file_name': 'query.emb',
    'output_file_name': 'para.index.part0_100',
    'test_data_cnt': 100, # should match with len of test_set
    'q_max_seq_len': 32,
    'p_max_seq_len': 128,
    'vocab_path': 'rocket_qa/model/config/vocab.txt',
    'ernie_config_path': 'rocket_qa/model/config/base/ernie_config.json',
    'label_map_config': None,
    'do_lower_case': True,
    'in_tokens': False,
    'random_seed': 1,
    'tokenizer': 'FullTokenizer',
    'for_cn': False,
    'task_id': 0,
    'predict_batch_size': None
}

Args = namedtuple('Args',args)
args = Args(**args)

inference_de.main(args)
