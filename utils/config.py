import argparse


# -*- coding: utf-8 -*-
# 导入必备的工具包
import torch
# 导入Vocabulary，目的：用于构建, 存储和使用 `str` 到 `int` 的一一映射
from fastNLP import Vocabulary
from transformers import BertTokenizer, AdamW
import json
import os, pathlib
from transformers import T5Model, BartModel
from transformers import T5Config, T5Tokenizer,  BartConfig, BartTokenizer

MODEL_CLASSES = {
    't5': (T5Config, T5Model, T5Tokenizer),
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
        self.train_data = data_dir + '/train_demo.json'
        self.dev_data = data_dir + '/dev.json'
        self.test_data = data_dir + '/test.json'
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

        self.save_path = self.output_dir + self.dataset + os.sep + self.model_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # print('模型保存路径:', sub_path)
        self.trained_model_path = self.save_path + os.sep + self.model_name + '_epochs_' + str(self.epochs) + '_bs_' + str(
            self.batch_size) + '_lr_' + str(
            self.learning_rate) + '_'
        # print('trained_model_path:', self.trained_model_path)




# 初始化参数
def init_argument():
    parser = argparse.ArgumentParser(description='智能创作平台标题摘要生成模型')

    parser.add_argument('--train_data', default='./data/train.json', help='训练数据集路径')
    parser.add_argument('--dev_data', default='./data/dev.json', help='验证数据集路径')
    parser.add_argument('--model_title', default='./saved_model/model_title', help='标题生成的模型参数')
    parser.add_argument('--model_summary', default='./saved_model/model_summary', help='摘要生成的模型参数')
    parser.add_argument('--pretrain_model', default='./t5_pegasus_pretrain', help='模型的预训练参数路径')
    parser.add_argument('--model_dir', default='./saved_model', help='模型参数的保存路径')
    parser.add_argument('--device', default='0', help='默认使用的gpu编号')

    parser.add_argument('--num_epoch', default=50, help='训练的epoch轮数')
    parser.add_argument('--batch_size', default=8, help='batch size')
    parser.add_argument('--lr', default=2e-4, help='学习率')
    parser.add_argument('--data_parallel', default=False, help='是否使用多卡训练')
    parser.add_argument('--max_len', default=1024, help='输入的最大长度')
    parser.add_argument('--max_len_generate', default=80, help='输出的最大长度')

    args = parser.parse_args()
    return args
