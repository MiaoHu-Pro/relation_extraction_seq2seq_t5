import argparse


# -*- coding: utf-8 -*-
# 导入必备的工具包
import torch
# 导入Vocabulary，目的：用于构建, 存储和使用 `str` 到 `int` 的一一映射
from fastNLP import Vocabulary
from transformers import BertTokenizer, AdamW
import json
import os, pathlib
from transformers import T5Model, T5ForConditionalGeneration, BartModel
from transformers import T5Config, T5Tokenizer,  BartConfig, BartTokenizer

MODEL_CLASSES = {
    't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
    'bart': (BartConfig, BartModel, BartTokenizer),  # BartModel 1024
}


class Config():
    def __init__(self, args=None):
        # 定义是否需要使用GPU
        self.args = args

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if '/' in args.model_name_or_path:
            self.model_name = args.model_name_or_path.split('/')[-1]
        else:
            self.model_name = args.model_name_or_path

        self.model_type = args.model_type.lower()
        print('model_type:', self.model_type)
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.model_type]
        _config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                               cache_dir=args.cache_dir if args.cache_dir else None)
        self.tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None)

        self.model = model_class.from_pretrained(args.model_name_or_path,
                                                 from_tf=bool('.ckpt' in args.model_name_or_path),
                                                 config=_config,
                                                 cache_dir=args.cache_dir if args.cache_dir else None)

        self.tokenizer = add_structural_tokens(self.tokenizer, self.model)

        self.output_dir = args.output_dir

        # 定义关系类别数
        # self.rel_class = args.rel_class
        # 定义一个批次多少样本
        self.batch_size = args.batch_size
        self.data_base_dir = args.data_base_dir
        self.dataset = args.dataset
        self.max_len = args.max_len
        self.max_len_generate = args.max_len_generate

        # 定义数据路径
        here = pathlib.Path(__file__).resolve().parent
        data_dir = os.path.join(here.parent, "data") + os.sep + args.dataset
        if args.do_develop:
            print('development environment, using train_demo.json for training and dev.json for validation')
            self.train_data = data_dir + '/train_demo.json'
            self.test_data = data_dir + '/test_demo.json'
            self.dev_data = data_dir + '/dev_demo.json'
        else:
            print('using train.json for training ')
            self.train_data = data_dir + '/train.json'
            self.test_data = data_dir + '/test.json'
            self.dev_data = data_dir + '/dev.json'

        self.relation_data = data_dir + '/relation.json'
        self.id2rel = json.load(open(self.relation_data, encoding='utf-8'))
        self.rel_vocab = Vocabulary(padding=None, unknown=None)
        self.rel_vocab.add_word_lst(list(self.id2rel.values()))
        self.data_parallel = args.data_parallel
        self.learning_rate = args.learning_rate
        self.hidden_dim = args.hidden_dim
        self.epochs = args.epochs
        self.dropout_prob = args.hidden_dropout_prob
        self.accumulation_steps = args.gradient_accumulation_steps

        self.save_path = self.output_dir + self.dataset+'_t5_re' + os.sep + self.model_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # print('模型保存路径:', sub_path)
        self.trained_model_path = self.save_path + os.sep + self.model_name + '_epochs_' + str(self.epochs) + '_bs_' + str(
            self.batch_size) + '_lr_' + str(
            self.learning_rate) + '_'
        # print('trained_model_path:', self.trained_model_path)

def add_structural_tokens(tokenizer, model=None):
    # Define them as special tokens, not just “normal” tokens
    special_tokens_dict = {
        "additional_special_tokens": ["<SUBJ>", "<PRED>", "<OBJ>", "<TRIPLE>"]
    }

    # Add only the missing ones
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    if num_added > 0:
        print(f"Added {num_added} structural special tokens: {special_tokens_dict['additional_special_tokens']}")
        if model is not None:
            model.resize_token_embeddings(len(tokenizer))
            print("Resized model embeddings to match new vocab size.")

    return tokenizer


