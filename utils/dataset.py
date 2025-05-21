import re
import time
from functools import partial

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json

from utils.utilities import get_args


# 使用分词器对数据集编码
def create_data(data, tokenizer, max_len=512, term='train'):
    datasets = []
    for title, content in data:
        features = {}
        text_ids = tokenizer.encode(content, max_length=max_len, truncation='only_first')
        # 如果是训练集，对标题也进行编码
        if term == 'train':
            summary_ids = tokenizer.encode(title, max_length=max_len, truncation='only_first')
            features = {'input_ids': text_ids,
                        'decoder_input_ids': summary_ids,
                        'attention_mask': [1] * len(text_ids),
                        'decoder_attention_mask': [1] * len(summary_ids)
                        }
        # 如果是测试集，只对文章内容进行编码
        elif term == 'dev':
            features = {'input_ids': text_ids,
                        'attention_mask': [1] * len(text_ids),
                        'title': title
                        }
        datasets.append(features)

    return datasets


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')


def collate_fn_train(data, conf):
    # "spo_list": [
    #     {"subject": "Cambodia", "object": "Penh", "predicate": "/location/country/administrative_divisions"},
    #     {"subject": "Cambodia", "object": "Penh", "predicate": "/location/location/contains"},
    #     {"subject": "Penh", "object": "Cambodia", "predicate": "/location/administrative_division/country"},
    #     {"subject": "Cambodia", "object": "Penh", "predicate": "/location/country/capital"}]

    text_list = [value[0] for value in data]
    triple_list = [value[1] for value in data]
    triples_token = [
        " ; ".join(
            f"{t['subject']}' {conf.tokenizer.eos_token }  "
            f"{t['predicate'].strip('/').replace('/', ' ').replace('_', ' ')}  "
            f" {conf.tokenizer.eos_token} {t['object']}"
            for t in example_spo_list
        )
        for example_spo_list in triple_list
    ]


    # triples_token = [
    #     " ; ".join(
    #         f"{t['subject']} has a relation of {t['predicate']} with {t['object']}"
    #         for t in example_spo_list
    #     )
    #     for example_spo_list in triple_list
    # ]

    # Encode the text list with padding
    text = conf.tokenizer(text_list, max_length=conf.max_len, padding="longest")
    triples = conf.tokenizer(triples_token, max_length=conf.max_len, padding="longest")

    features = {'input_ids': torch.tensor(text['input_ids']).to(conf.device),
                'decoder_input_ids': torch.tensor(triples['input_ids']).to(conf.device),
                'attention_mask': torch.tensor(text['attention_mask']).to(conf.device),
                'decoder_attention_mask': torch.tensor(triples['attention_mask']).to(conf.device),
                }

    # print(f"features: {features}")

    return features


def collate_fn_dev(data, conf):
    text_list = [value[0] for value in data]
    triple_list = [value[1] for value in data]

    triples_token = [
        " ; ".join(
            f"{t['subject']}' {conf.tokenizer.eos_token }  "
            f"{t['predicate'].strip('/').replace('/', ' ').replace('_', ' ')}  "
            f" {conf.tokenizer.eos_token} {t['object']}"
            for t in example_spo_list
        )
        for example_spo_list in triple_list
    ]

    # Encode the text list with padding
    text = conf.tokenizer(text_list, max_length=conf.max_len, padding=True)

    features = {'input_ids': torch.tensor(text['input_ids']).to(conf.device),
                'attention_mask': torch.tensor(text['attention_mask']).to(conf.device),
                'text_list': text_list,
                'triples_token': triples_token,
                }

    return features




def print_row(text, triples):
    print("=" * 20)
    print("text", text)
    print("spo_list", triples)
    print("=" * 20)


class MyDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.dataset = [json.loads(line) for line in open(data_path, encoding='utf8')]
        print("dataset len", len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        content = self.dataset[item]
        text = content['text']
        spo_list = content['spo_list']

        return text, spo_list

    def get_text(self, item):
        content = self.dataset[item]
        text = content['text']
        spo_list = content['spo_list']
        triples = " ; ".join(
            f"{t['subject']} has a relation of {t['predicate']} with {t['object']}"
            for t in spo_list
        )
        # print("=" * 20)
        # print_row(text, triples)
        return text, triples


def prepare_data(conf):
    # 实例化训练集dataset
    # print("perpare_data")
    # print(conf.train_data)
    train_dataset = MyDataset(conf.train_data)
    # for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
    #     # 打印训练集的前20条数据
    #     print("train_dataset",  )
    #
    #     train_dataset.get_text(i)

    # 实例化验证集dataset
    dev_dataset = MyDataset(conf.dev_data)
    # 实例化测试集dataset
    test_dataset = MyDataset(conf.test_data)

    # 实例化训练数据集的dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=conf.batch_size,
                                  shuffle=True,
                                  collate_fn=partial(collate_fn_train, conf=conf),
                                  drop_last=True)

    # 实例化验证数据集的dataloader
    dev_dataloader = DataLoader(dataset=dev_dataset,
                                batch_size=conf.batch_size,
                                shuffle=True,
                                collate_fn=partial(collate_fn_dev, conf=conf),
                                drop_last=True)

    # 实例化测试数据集的dataloader
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=conf.batch_size,
                                 shuffle=True,
                                 collate_fn=partial(collate_fn_dev, conf=conf),
                                 drop_last=True)

    return train_dataloader, dev_dataloader, test_dataloader


if __name__ == '__main__':
    args = get_args()

    from config import Config

    config = Config(args)

    train_dataloader, dev_dataloader, test_dataloader = prepare_data(config)

    # python3 dataset.py --dataset=nyt  --data_base_dir=./data   --model_name_or_path=t5-base   --output_dir=./save_model --do_train  --model_type=t5 --epochs 1
