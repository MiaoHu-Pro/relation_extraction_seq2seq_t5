
import pandas as pd
from tqdm import tqdm
import torch
from utils.compute_rouge import *

def model2test(model, config, test_dataloader):
    # 设置模型为验证模式
    model.eval()
    gen_list = []
    label_list = []

    with torch.no_grad():
        for item in tqdm(test_dataloader, desc='\ntest'):
            content = {
                k: v.to(config.device)
                for k, v in item.items() if k != 'triples_token' and k != 'text_list'

            }
            triples_token = item['triples_token']  # now a single string
            input_text = item['text_list']  # now a single string

            if config.data_parallel and torch.cuda.is_available():
                gen = model.module.generate(
                    eos_token_id=config.tokenizer.sep_token_id,
                    decoder_start_token_id=config.tokenizer.cls_token_id,
                    **content)
            else:  # single gpu
                gen = model.generate(
                    max_length=conf.max_triple_len,  # e.g. 64 or 128
                    eos_token_id=conf.tokenizer.eos_token_id,
                    pad_token_id=conf.tokenizer.pad_token_id,
                    early_stopping=True,


                    decoder_start_token_id=config.tokenizer.cls_token_id,
                    **content)
            gen = config.tokenizer.batch_decode(gen, skip_special_tokens=True)

            gen_list.extend(gen)

            with open(config.trained_model_path + 'test_gen_log.txt', 'a') as f:
                f.write(str(input_text) + ' # ' + str(triples_token) + ' # ' + str(gen) + '\n')

                f.close()

            label_list.extend(triples_token)
        # 计算生成标题的平均rouge分数
        scores = compute_rouges(gen_list, label_list, 'f')
        print("验证集的rouge分数为: {}".format(scores))
        # 单独保存rouge-l分数最高的模型
        rouge_l = scores['rouge-l']



