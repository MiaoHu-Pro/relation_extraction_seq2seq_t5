# coding:utf-8
import torch

from utils.config import init_argument
from utils.compute_rouge import compute_rouge


def generate(args, device, tokenizer, model, content):
    # 设置模型为验证模式
    model.eval()
    # 使用分词器对输入文本编码
    text_ids = tokenizer.encode(content, max_length=args.max_len, truncation='only_first')
    # 组装预测数据
    text_ids = torch.tensor([text_ids]).to(device)
    attention_mask = [1] * len(text_ids)
    attention_mask = torch.tensor([attention_mask]).to(device)
    input = {'input_ids': text_ids, 'attention_mask': attention_mask}
    # 调用模型得到预测输出
    output = model.generate(max_length=200,
                            eos_token_id=tokenizer.sep_token_id,
                            decoder_start_token_id=tokenizer.cls_token_id,
                            **input)
    # 使用分词器对输出解码
    output = tokenizer.batch_decode(output, skip_special_tokens=True)
    output = output[0].replace(' ', '')
    return output


