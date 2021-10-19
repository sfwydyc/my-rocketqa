#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning on classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import json
import multiprocessing
import numpy as np

# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import paddle.fluid as fluid

from .reader import reader_de_predict
from .model.ernie import ErnieConfig
from .finetune.dual_encoder_predict import create_model, predict
from .utils.args import print_arguments, check_cuda, prepare_logger
from .utils.init import init_pretraining_params, init_checkpoint
from .finetune_args import parser
from pathlib import PurePath


class DEPredictor(object):

    def __init__(self, conf_path, use_cuda, gpu_card_id, batch_size, cls_type):
        args = self._parse_args(conf_path)
        args.use_cuda = use_cuda
        ernie_config = ErnieConfig(args.ernie_config_path)
        ernie_config.print_config()
        self.batch_size = batch_size
        self.cls_type = cls_type

        if args.use_cuda:
            dev_list = fluid.cuda_places()
            place = dev_list[gpu_card_id]
            dev_count = len(dev_list)
        else:
            place = fluid.CPUPlace()
            dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
        self.exe = fluid.Executor(place)

        self.reader = reader_de_predict.DEPredictorReader(
            vocab_path=args.vocab_path,
            label_map_config=args.label_map_config,
            q_max_seq_len=args.q_max_seq_len,
            p_max_seq_len=args.p_max_seq_len,
            do_lower_case=args.do_lower_case,
            in_tokens=args.in_tokens,
            random_seed=args.random_seed,
            tokenizer=args.tokenizer,
            for_cn=args.for_cn,
            task_id=args.task_id)

        startup_prog = fluid.Program()

        self.test_prog = fluid.Program()
        with fluid.program_guard(self.test_prog, startup_prog):
            with fluid.unique_name.guard():
                self.test_pyreader, self.graph_vars = create_model(
                    args,
                    pyreader_name='test_reader',
                    ernie_config=ernie_config,
                    is_prediction=True)

        self.test_prog = self.test_prog.clone(for_test=True)

        self.exe = fluid.Executor(place)
        self.exe.run(startup_prog)

        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                            "only doing validation or testing!")
        init_checkpoint(
            self.exe,
            args.init_checkpoint,
            main_program=startup_prog)

    def _parse_args(self, conf_path):
        args = parser.parse_args()
        try:
            with open(conf_path, 'r', encoding='utf8') as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing model config file '%s'" %conf_path)

        args.do_train = False
        args.do_val = False
        args.do_test = True
        args.use_fast_executor = True
        args.q_max_seq_len = config_dict['q_max_seq_len']
        args.p_max_seq_len = config_dict['p_max_seq_len']
        args.ernie_config_path = config_dict['model_conf_path']
        args.vocab_path = config_dict['model_vocab_path']
        args.init_checkpoint = config_dict['model_checkpoint_path']

        return args


    def get_cls_feats(self, data):

        self.test_pyreader.decorate_tensor_provider(
            self.reader.data_generator(data, self.batch_size, shuffle=False))

        self.test_pyreader.start()
        fetch_list = [self.graph_vars["q_rep"].name, self.graph_vars["p_rep"].name]
        embs = []

        while True:
            try:

                q_rep, p_rep = self.exe.run(program=self.test_prog,
                                                fetch_list=fetch_list)

                if self.cls_type == 'query':
                    embs.append(q_rep)
                elif self.cls_type == 'para':
                    embs.append(p_rep)

            except fluid.core.EOFException:
                self.test_pyreader.reset()
                break

        return np.concatenate(embs)[:len(data)]

