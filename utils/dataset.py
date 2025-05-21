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

def get_input(config, sample):
    """
     data: list of (text_str, spo_list) for each example in the dev set.
     returns:
       - input_ids, attention_mask  → for model.generate(...)
       - text_list, targets_str     → for metric computation
     """
    text_list = [sample]

    # tokenize the source texts
    src = config.tokenizer(
        text_list,
        max_length=config.max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"  # you’ll get torch tensors of shape [1, seq_len]
    )

    # wrap to tensors
    input_ids = torch.tensor(src["input_ids"]).to(config.device)
    attention_mask = torch.tensor(src["attention_mask"]).to(config.device)

    data_fetures = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "text_list": text_list,  # raw inputs, for logging if you like
    }
    # print("data_fetures===> ", data_fetures)
    return data_fetures



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


import re


def clean_predicate(p: str) -> str:
    # 1) strip leading/trailing slashes/underscores
    p = p.strip("/_")
    # 2) replace any remaining slash or underscore with a space
    p = re.sub(r"[\/_]+", " ", p)
    # 3) collapse multiple spaces
    return re.sub(r"\s+", " ", p).strip()


def collate_fn_train(data, conf):
    # "spo_list": [
    #     {"subject": "Cambodia", "object": "Penh", "predicate": "/location/country/administrative_divisions"},
    #     {"subject": "Cambodia", "object": "Penh", "predicate": "/location/location/contains"},
    #     {"subject": "Penh", "object": "Cambodia", "predicate": "/location/administrative_division/country"},
    #     {"subject": "Cambodia", "object": "Penh", "predicate": "/location/country/capital"}]

    text_list = [ex[0] for ex in data]
    spo_lists = [ex[1] for ex in data]

    SEP = "<TRIPLE>"
    SUB = "<SUBJ>"
    PRED = "<PRED>"
    OBJ = "<OBJ>"

    targets = []
    for spo_list in spo_lists:
        pieces = []
        for spo in spo_list:
            s = spo["subject"]
            p = clean_predicate(spo["predicate"])
            o = spo["object"]
            pieces.append(f"{SUB} {s} {PRED} {p} {OBJ} {o}")
        # join the per-spo pieces with your triple separator
        targets.append(f" {SEP} ".join(pieces))

    # print("targets===> ", targets)
    # tokenize source and target
    src = conf.tokenizer(text_list,
                         max_length=conf.max_len,
                         padding="longest",
                         truncation=True)

    tgt = conf.tokenizer(targets,
                         max_length=conf.max_len,
                         padding="longest",
                         truncation=True)

    return {
        "input_ids": torch.tensor(src["input_ids"]).to(conf.device),
        "attention_mask": torch.tensor(src["attention_mask"]).to(conf.device),
        "decoder_input_ids": torch.tensor(tgt["input_ids"]).to(conf.device),
        "decoder_attention_mask": torch.tensor(tgt["attention_mask"]).to(conf.device),
    }

    # text_list = [value[0] for value in data]
    # spo_lists = [value[1] for value in data]
    #
    # sep_tok = "<TRIPLE>"
    # subj_tok = "<SUBJ>"
    # pred_tok = "<PRED>"
    # obj_tok = "<OBJ>"
    #
    # targets = []
    # for spo_list in spo_lists:
    #     # each spo_list is a list of dicts with keys: subject, predicate, object
    #     triples = []
    #     for spo in spo_list:
    #         s = spo["subject"]
    #         p = spo["predicate"]
    #         o = spo["object"]
    #         triples.append(f"{subj_tok} {s} {pred_tok} {p} {obj_tok} {o}")
    #     # join multiple triples with the sep token
    #     targets.append(f" {sep_tok} ".join(triples))
    #
    # # print("train targets: ", targets)
    #
    # # Encode the text list with padding
    # text = conf.tokenizer(text_list, max_length=conf.max_len, padding="longest")
    # triples = conf.tokenizer(targets, max_length=conf.max_len, padding="longest")
    #
    # features = {'input_ids': torch.tensor(text['input_ids']).to(conf.device),
    #             'decoder_input_ids': torch.tensor(triples['input_ids']).to(conf.device),
    #             'attention_mask': torch.tensor(text['attention_mask']).to(conf.device),
    #             'decoder_attention_mask': torch.tensor(triples['attention_mask']).to(conf.device),
    #             }
    #
    #
    #
    # return features


def spo_list_to_triple_string(spo_list):
    SEP = "<TRIPLE>"
    SUB = "<SUBJ>"
    PRED = "<PRED>"
    OBJ = "<OBJ>"
    pieces = []
    for spo in spo_list:
        s = spo["subject"]
        p = clean_predicate(spo["predicate"])
        o = spo["object"]
        pieces.append(f"{SUB} {s} {PRED} {p} {OBJ} {o}")
    return f" {SEP} ".join(pieces)


def collate_fn_dev(data, conf):
    """
    data: list of (text_str, spo_list) for each example in the dev set.
    returns:
      - input_ids, attention_mask  → for model.generate(...)
      - text_list, targets_str     → for metric computation
    """
    text_list = [ex[0] for ex in data]
    spo_lists = [ex[1] for ex in data]

    # build raw target strings
    targets_str = [spo_list_to_triple_string(spo_list)
                   for spo_list in spo_lists]

    # tokenize the source texts
    src = conf.tokenizer(
        text_list,
        max_length=conf.max_len,
        padding="longest",
        truncation=True,
        return_tensors=None  # we’ll wrap to tensors ourselves
    )

    # wrap to tensors
    input_ids = torch.tensor(src["input_ids"]).to(conf.device)
    attention_mask = torch.tensor(src["attention_mask"]).to(conf.device)

    data_fetures = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "text_list": text_list,  # raw inputs, for logging if you like
        "triples_token": targets_str,  # raw gold triple strings, for metrics
    }
    # print("data_fetures===> ", data_fetures)
    return data_fetures

    # text_list = [value[0] for value in data]
    # spo_lists = [value[1] for value in data]
    #
    # triples_token = []
    # sep_tok = "<TRIPLE>"
    # subj_tok = "<SUBJ>"
    # pred_tok = "<PRED>"
    # obj_tok = "<OBJ>"
    # for spo_list in spo_lists:
    #     # each spo_list is a list of dicts with keys: subject, predicate, object
    #     triples = []
    #     for spo in spo_list:
    #         s = spo["subject"]
    #         p = spo["predicate"]
    #         o = spo["object"]
    #         triples.append(f"{subj_tok} {s} {pred_tok} {p} {obj_tok} {o}")
    #     # join multiple triples with the sep token
    #     triples_token.append(f" {sep_tok} ".join(triples))
    #
    # text = conf.tokenizer(text_list, max_length=conf.max_len, padding=True)
    # dec = conf.tokenizer(
    #     triples_token,
    #     max_length=conf.max_len,
    #     padding=True)
    # features = {'input_ids': torch.tensor(text['input_ids']).to(conf.device),
    #             'attention_mask': torch.tensor(text['attention_mask']).to(conf.device),
    #
    #
    #             'text_list': text_list,
    #             'triples_token': triples_token,
    #             }
    #
    # # print("features: ", features)
    #
    # return features


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
