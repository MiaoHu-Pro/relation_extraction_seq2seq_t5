import pandas as pd
from tqdm import tqdm
import torch
from utils.compute_rouge import *
from utils.utilities import parse_predictions


def model2test(model, config, test_dataloader):

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
                    max_new_tokens=config.max_len_generate,
                    eos_token_id=config.tokenizer.eos_token_id,
                    pad_token_id=config.tokenizer.pad_token_id,
                    **content)
            else:  # single gpu
                gen = model.generate(
                    max_new_tokens=config.max_len_generate,
                    eos_token_id=config.tokenizer.eos_token_id,
                    pad_token_id=config.tokenizer.pad_token_id,
                    **content)
            gen = config.tokenizer.batch_decode(gen, skip_special_tokens=False)

            # print("original triples token:", triples_token)
            # print("generated:", gen)
            triples = parse_predictions(gen)
            # print("parsed generated:", triples)

            gen_list.extend(triples)
            label_list.extend(triples_token)

            with open(config.trained_model_path + 'test_gen_log.txt', 'a') as f:
                num = len(input_text)
                for i in range(num):
                    f.write(str(input_text[i]) + ' ## ' + str(triples_token[i]) + ' ## ' + str(triples[i]) + '\n')

                f.close()


        # 计算生成标题的平均rouge分数
        scores = compute_rouges(gen_list, label_list, 'f')
        print("验证集的rouge分数为: {}".format(scores))
        # 单独保存rouge-l分数最高的模型
        rouge_l = scores['rouge-l']

        return scores
