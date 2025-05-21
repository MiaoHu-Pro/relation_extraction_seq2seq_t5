import logging

import torch
from transformers import MT5ForConditionalGeneration
from utils.config import init_argument
from utils.compute_rouge import *
from utils.train_model import train_model
from model.Text2TripleModel import *

import pandas as pd
import torch
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)


def model2train(model, conf, train_iter, dev_iter, optimizer):
    args = conf.args
    epochs = conf.epochs
    best_triple_f1 = 0
    for epoch in range(epochs):
        train_epoch(model, conf, train_iter, dev_iter, optimizer, best_triple_f1, epoch)
    torch.save(model.state_dict(), conf.trained_model_path + 'last_model.pth')


# 定义内部迭代函数:train_epoch
def train_epoch(model, conf, train_iter, dev_iter, optimizer, best_triple_f1, epoch):
    print('=' * 40)
    loss = 0
    model.train()
    for i, item in enumerate(tqdm(train_iter, desc='Epoch {}\n'.format(epoch))):

        prob = model(**item)[0]

        mask = item['decoder_attention_mask'][:, 1:].reshape(-1).bool()
        prob = prob[:, :-1]
        prob = prob.reshape((-1, prob.size(-1)))[mask]
        labels = item['decoder_input_ids'][:, 1:].reshape(-1)[mask]
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(prob, labels)
        if i % 100 == 0:
            logger.info("Iter {}:  Training Loss: {}".format(i, loss.item()))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(epoch, "is completed.")
    print('*' * 80)
    torch.save(model.state_dict(), conf.trained_model_path + 'epoch_%s_model.pth' % epoch)
    results = model2dev(model, conf, epoch, dev_iter)
    rouge_l = results['rouge-l']
    # Append evaluation log
    log_line = ('epoch:{},'
                'rouge-l:{:.4f}, '
                'rouge-2:{:.4f}, '
                'train loss:{:.4f}'.format(epoch,
                                           results['rouge-l'],
                                           results['rouge-2'],
                                           loss.item()))

    eval_log = conf.trained_model_path + 'trained_model_evaluation_log.txt'
    with open(eval_log, 'a') as f:
        f.write(log_line + '\n')
    f.close()

    print(rouge_l)
    if rouge_l > best_triple_f1:
        best_triple_f1 = rouge_l
        torch.save(model.state_dict(), conf.trained_model_path + 'best_f1.pth')

    return best_triple_f1


# 定义验证函数
def model2dev(model, conf, epoch, dev_iter):
    print('进入验证循环')
    # 验证模型性能，将分数最高的模型单独保存
    model.eval()
    gen_titles = []
    label_titles = []

    for item in tqdm(dev_iter, desc='valid'):

        # batch is a List[Dict], so you need another loop
        # print("item:", item)
        content = {
            k: v
            for k, v in item.items() if k != 'triples_token' and k != 'text_list'

        }
        triples_token = item['triples_token']  # now a single string
        input_text = item['text_list']  # now a single string

        if conf.data_parallel and torch.cuda.is_available():
            gen = model.module.generate(
                max_new_tokens=conf.max_len_generate,
                eos_token_id=conf.tokenizer.sep_token_id,
                decoder_start_token_id=conf.tokenizer.cls_token_id,
                **content)
        else:  # single gpu
            gen = model.generate(
                max_new_tokens=conf.max_len_generate,
                eos_token_id=conf.tokenizer.eos_token_id,
                pad_token_id=conf.tokenizer.pad_token_id,
                **content)
        gen = conf.tokenizer.batch_decode(gen, skip_special_tokens=False)

        gen_titles.extend(gen)

        with open(conf.trained_model_path + "epoch_" + str(epoch) + '_gen_log.txt', 'a') as f:
            f.write(str(input_text) + ' # ' + str(triples_token) + ' # ' + str(gen) + '\n')
            f.close()
        label_titles.extend(triples_token)
    # 计算生成标题的平均rouge分数
    scores = compute_rouges(gen_titles, label_titles, 'f')
    print("验证集的rouge分数为: {}".format(scores))
    # 单独保存rouge-l分数最高的模型
    rouge_l = scores['rouge-l']

    return scores
