import torch
from transformers import MT5ForConditionalGeneration
from utils.config import init_argument
from utils.tokenizer import T5PegasusTokenizer
from utils.dataset import prepare_data
from utils.train_model import train_model


# -*- coding: utf-8 -*-

from model.Text2TripleModel import *
from utils.process import convert_score_to_zero_one, extract_sub, extract_obj_and_rel

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
    my_loss = 0

    for step, (inputs, labels) in enumerate(tqdm(train_iter, desc="Training\n")):
        # 将数据送入模型
        logist = model(**inputs)
        # 计算损失
        my_loss = model.compute_loss(**logist, **labels)
        # 梯度清零
        model.zero_grad()
        # optimizer.zero_grad()
        # 反向传播
        my_loss.backward()
        # 梯度更新
        optimizer.step()
        model.zero_grad()

    print(epoch, "is completed.")
    print('*' * 80)
    torch.save(model.state_dict(), conf.trained_model_path + 'epoch_%s_model.pth' % epoch)
    results = model2dev(model, conf, dev_iter)

    # Append evaluation log
    log_line = ('epoch:{},'
                'sub_precision:{:.4f}, '
                'sub_recall:{:.4f}, '
                'sub_f1:{:.4f}, '
                'triple_precision:{:.4f}, '
                'triple_recall:{:.4f}, '
                'triple_f1:{:.4f},'
                'train loss:{:.4f}'.format(epoch,
                                           results[0],
                                           results[1],
                                           results[2],
                                           results[3],
                                           results[4],
                                           results[5],
                                           my_loss.item()))

    eval_log = conf.trained_model_path + 'trained_model_evaluation_log.txt'
    with open(eval_log, 'a') as f:
        f.write(log_line + '\n')
    f.close()

    print(results[-2])
    if results[-2] > best_triple_f1:
        best_triple_f1 = results[-2]
        torch.save(model.state_dict(), conf.trained_model_path + 'best_f1.pth')

    return best_triple_f1


# 定义验证函数
def model2dev(model, conf, dev_iter):
    print('进入验证循环')
    model.eval()
    # 定义一个df，来展示模型的指标。
    df = pd.DataFrame(columns=['TP', 'PRED', "REAL", 'p', 'r', 'f1'], index=['sub', 'triple'])
    df.fillna(0, inplace=True)
    for inputs, labels in tqdm(dev_iter):
        logits = model(**inputs)
        pred_sub_heads = convert_score_to_zero_one(logits['pred_sub_heads'])
        pred_sub_tails = convert_score_to_zero_one(logits['pred_sub_tails'])
        sub_heads = convert_score_to_zero_one(labels['sub_heads'])
        sub_tails = convert_score_to_zero_one(labels['sub_tails'])
        batch_size = inputs['input_ids'].shape[0]
        obj_heads = convert_score_to_zero_one(labels['obj_heads'])
        obj_tails = convert_score_to_zero_one(labels['obj_tails'])
        pred_obj_heads = convert_score_to_zero_one(logits['pred_obj_heads'])
        pred_obj_tails = convert_score_to_zero_one(logits['pred_obj_tails'])

        # pred_sub_heads形状--》[8, 70, 1],pred_sub_heads[batch_index]形状：【70, 1】
        # pred_sub_heads[batch_index].squeeze()形状：【70】
        # pred_sub_tails[batch_index].squeeze()形状：【70】

        # pred_obj_heads形状--》【8,70,18】,pred_obj_heads[batch_index]形状：【70，18】

        for batch_index in range(batch_size):
            pred_subs = extract_sub(conf, pred_sub_heads[batch_index].squeeze(1),
                                    pred_sub_tails[batch_index].squeeze(1))
            # print(f'pred_subs-->{pred_subs}')
            true_subs = extract_sub(conf, sub_heads[batch_index].squeeze(), sub_tails[batch_index].squeeze())
            # print(f'true_subs-->{true_subs}')
            pred_ojbs = extract_obj_and_rel(conf, pred_obj_heads[batch_index], pred_obj_tails[batch_index])
            # print(f'pred_ojbs-->{pred_ojbs}')
            true_objs = extract_obj_and_rel(conf, obj_heads[batch_index], obj_tails[batch_index])
            df['PRED']['sub'] += len(pred_subs)

            df['REAL']['sub'] += len(true_subs)

            for true_sub in true_subs:
                if true_sub in pred_subs:
                    df['TP']['sub'] += 1

            df['PRED']['triple'] += len(pred_ojbs)

            df['REAL']['triple'] += len(true_objs)

            for true_obj in true_objs:
                if true_obj in pred_ojbs:
                    df['TP']['triple'] += 1

    df.loc['sub', 'p'] = df['TP']['sub'] / (df['PRED']['sub'] + 1e-9)
    df.loc['sub', 'r'] = df['TP']['sub'] / (df['REAL']['sub'] + 1e-9)
    df.loc['sub', 'f1'] = 2 * df['p']['sub'] * df['r']['sub'] / (df['p']['sub'] + df['r']['sub'] + 1e-9)
    sub_precision = df['TP']['sub'] / (df['PRED']['sub'] + 1e-9)
    sub_recall = df['TP']['sub'] / (df['REAL']['sub'] + 1e-9)
    sub_f1 = 2 * sub_precision * sub_recall / (sub_precision + sub_recall + 1e-9)

    df.loc['triple', 'p'] = df['TP']['triple'] / (df['PRED']['triple'] + 1e-9)
    df.loc['triple', 'r'] = df['TP']['triple'] / (df['REAL']['triple'] + 1e-9)
    df.loc['triple', 'f1'] = 2 * df['p']['triple'] * df['r']['triple'] / (
            df['p']['triple'] + df['r']['triple'] + 1e-9)

    triple_precision = df['TP']['triple'] / (df['PRED']['triple'] + 1e-9)
    triple_recall = df['TP']['triple'] / (df['REAL']['triple'] + 1e-9)
    triple_f1 = 2 * triple_precision * triple_recall / (
            triple_precision + triple_recall + 1e-9)

    return sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df


