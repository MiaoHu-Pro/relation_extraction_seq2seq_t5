from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import pickle
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from predict import model2test
from utils.config import Config
from train import model2train
from utils.dataset import prepare_data
from utils.utilities import get_args
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
# from model.Text2TripleModel import Text2TripleT5, Text2TripleBart
from transformers import T5Model, BartModel

MODELS = {
    't5': T5Model,
    'bart': BartModel,
}

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)  # Only show errors, not warnings


def main():
    args = get_args()

    # args.rel_class = get_rel_class(args.dataset)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # config
    config = Config(args)
    from utils.utilities import print_and_save_params

    print_and_save_params(config, config.trained_model_path)

    train_dataloader, dev_dataloader, test_dataloader = prepare_data(config)

    if args.do_train:
        run_train(config, train_dataloader, dev_dataloader)

    # Test
    if args.do_test:
        run_test(config, test_dataloader)


def run_train(config, train_dataloader, dev_dataloader):
    # model = MODELS[config.model_type](conf=config).to(config.device)
    # model = MODELS[config.model_type].to(config.device)

    model = config.model.to(config.device)

    if config.data_parallel and torch.cuda.is_available():
        device_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    param_list = list(model.named_parameters())
    # logger.info(f'param_list-->{param_list}')
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_list if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_list if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=10e-8)

    # if config.args.max_steps > 0:
    #     t_total = config.args.max_steps
    #     config.args.epochs = config.args.max_steps // (
    #             len(train_dataloader) // config.args.gradient_accumulation_steps) + 1
    # else:
    #     t_total = len(train_dataloader) // config.args.gradient_accumulation_steps * config.args.epochs
    #
    # total_steps = len(train_dataloader) // config.args.gradient_accumulation_steps * config.args.epochs
    #
    # warmup_steps = max(1, int(config.args.warmup_proportion * total_steps))
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
    #                                             num_training_steps=t_total)

    model2train(model, config, train_dataloader, dev_dataloader, optimizer)


def run_test(config, test_dataloader):
    logger.info('do testing ...')
    # model = MODELS[config.model_type](conf=config).to(config.device)
    model = config.model.to(config.device)
    best_model_path = config.trained_model_path + 'best_f1.pth'
    last_model_path = config.trained_model_path + 'last_model.pth'

    # Check if best_f1.pth exists, otherwise load last_model.pth
    if os.path.isfile(best_model_path):
        # print   (f"Loading best model from {best_model_path}")
        logger.info(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
    elif os.path.isfile(last_model_path):
        # print  (f"Loading last model from {last_model_path}")
        logger.info(f"Loading last model from {last_model_path}")
        model.load_state_dict(torch.load(last_model_path))
    else:
        raise FileNotFoundError(f"Neither {best_model_path} nor {last_model_path} exists.")

    # 模型训练
    result = model2test(model, config, test_dataloader)
    logger.info('------------------test result------------------')
    logger.info(result)

    with open(config.trained_model_path + 'test_result.txt', model='a') as f:
        for key, value in result.items():
            f.write(f'{key}: {value}\n')
        f.close()


if __name__ == '__main__':
    main()

# python3 run.py --dataset=nyt  --data_base_dir=./data   --model_name_or_path=bert-base-cased   --output_dir=./save_model --do_train  --model_type=bert --epochs 1
#  python3 run.py --dataset=nyt  --data_base_dir=./data   --model_name_or_path=gpt2   --output_dir=./save_model/  --model_type gpt2  --do_train --do_test  --epochs 1
#  python3 run.py --dataset=nyt  --data_base_dir=./data   --model_name_or_path=bart   --output_dir=./save_model/  --model_type bart  --do_train --do_test  --epochs 1  --hidden_size 1024
#  python3 run.py --dataset=nyt  --data_base_dir=./data   --model_name_or_path=bart   --output_dir=./save_model/  --model_type bart  --do_train --do_test  --epochs 1  --hidden_dim  1024
#  python3 run.py --dataset=nyt  --data_base_dir=./data   --model_name_or_path=bart-base   --output_dir=./save_model/  --model_type bart  --do_train --do_test  --epochs 1  --hidden_dim  1024
#  python3 run.py --dataset=nyt  --data_base_dir=./data   --model_name_or_path=../bart   --output_dir=./save_model/  --model_type bart  --do_train --do_test  --epochs 1  --hidden_dim  1024
#  python3 run.py --dataset=nyt  --data_base_dir=./data   --model_name_or_path=../bart-base   --output_dir=./save_model/  --model_type bart  --do_train --do_test  --epochs 1  --hidden_dim  1024
#  python3 run.py --dataset=nyt  --data_base_dir=./data   --model_name_or_path=../t5-base   --output_dir=./save_model/  --model_type t5  --do_train --do_test  --epochs 1
#  python3 run.py --dataset=nyt  --data_base_dir=./data   --model_name_or_path=../bart-base   --output_dir=./save_model/  --model_type bart  --do_train --do_test  --epochs 1  --hidden_dim  1024

# this is origional data:
# {"text": "Israel has been carrying out ground raids in the West Bank and occasional airstrikes in Gaza , saying it is responding to planned or actual Palestinian attacks .",
#  "spo_list": [{"subject": "Israel", "object": "Bank", "predicate": "/location/location/contains"}]}
#
# so the input is a sentence -- Israel has been carrying out ground raids in the West Bank and occasional airstrikes in Gaza , saying it is responding to planned or actual Palestinian attacks .
#
# the output becames Israel has a relation of location location contains with Bank
#
# I would like to transfer the relation extraction to a sequence to sequence problem
#
# feed a sentence to model, then get a relational fact (then make it to a triple)

# Israel has been carrying out ground raids in the West Bank and occasional airstrikes in Gaza,
# saying it is responding to planned or actual Palestinian attacks.

# <SUBJ> Israel <PRED> /location/location/contains <OBJ> West Bank <TRIPLE> …



"""
["An article on Sept. 15 about the National Football League 's effort to
 create a home-field atmosphere for the New Orleans Saints misstated the name 
 of the New Orleans police superintendent , who was to take part in the coin 
 toss at a game with the New York Giants ."
 
    # ['<SUBJ> Saints <PRED> /sports/sports_team/location <OBJ> Orleans 
    <TRIPLE> <SUBJ> Orleans <PRED> /sports/sports_team_location/teams <OBJ> Saints', 
    
    
      
      # ['Saints <PRED> /sports/sports_team/location <OBJ> Orleans <TRIPLE> <SUBJ> Orleans', 
      
      



"""
