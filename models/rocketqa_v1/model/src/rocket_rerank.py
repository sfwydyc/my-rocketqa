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
import time
import logging
import multiprocessing

# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import paddle.fluid as fluid

from .reader import reader_ce as reader_ce
from .model.ernie import ErnieConfig
from .finetune.cross_encoder import create_model, evaluate, predict
from .optimization import optimization
from .utils.args import print_arguments, check_cuda, prepare_logger
from .utils.init import init_pretraining_params, init_checkpoint
from .finetune_args import parser

from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from pathlib import PurePath

args = parser.parse_args()
args.verbose = True
args.do_train = False
args.do_val = False
args.do_test = True
args.max_seq_len = 160
conf_path = PurePath(os.path.dirname(os.path.realpath(__file__))).parent / 'config'
args.vocab_path = str(conf_path / 'vocab.txt')
args.ernie_config_path = str(conf_path / 'large/ernie_config.json')
log = logging.getLogger()


class RocketReRanker(object):

    def __init__(self,use_cuda,batch_size,init_checkpoint_dir):
        args.init_checkpoint = init_checkpoint_dir
        ernie_config = ErnieConfig(args.ernie_config_path)
        ernie_config.print_config()

        if use_cuda:
            dev_list = fluid.cuda_places()
            place = dev_list[0]
            dev_count = len(dev_list)
        else:
            place = fluid.CPUPlace()
            dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
        self.exe = fluid.Executor(place)

        self.reader = reader_ce.ListClassifyReader(
            vocab_path=args.vocab_path,
            label_map_config=args.label_map_config,
            max_seq_len=args.max_seq_len,
            total_num=args.train_data_size,
            do_lower_case=args.do_lower_case,
            in_tokens=args.in_tokens,
            random_seed=args.random_seed,
            tokenizer=args.tokenizer,
            for_cn=args.for_cn,
            task_id=args.task_id)

        startup_prog = fluid.Program()
        if args.random_seed is not None:
            startup_prog.random_seed = args.random_seed

        args.predict_batch_size = batch_size

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
            


    def get_scores(self,data):

        self.test_pyreader.decorate_tensor_provider(
            self.reader.data_generator(
                data,
                batch_size=args.predict_batch_size,
                epoch=1,
                dev_count=1,
                shuffle=False))

        qids, preds, probs = predict(
            self.exe,
            self.test_prog,
            self.test_pyreader,
            self.graph_vars)

        return probs[:,1]