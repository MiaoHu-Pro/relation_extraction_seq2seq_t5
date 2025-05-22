# -*- coding: utf-8 -*-
import logging
import re
from test import *
from flask import Flask, render_template, request, jsonify, make_response
from utils.utilities import get_args, parse_predictions
from utils.dataset import get_input
from utils.config import *
app = Flask(__name__)

logger = logging.getLogger(__name__)




def model2predict(model, config, sample):
    print('model2predict', sample)
    new_dict = {}
    inputs = get_input(config, sample)

    content = {k: v for k, v in inputs.items() if k != 'text_list'}

    gen = model.generate(
        max_new_tokens=config.max_len_generate,
        eos_token_id=config.tokenizer.eos_token_id,
        pad_token_id=config.tokenizer.pad_token_id,
        **content)
    pre_triples = config.tokenizer.batch_decode(gen, skip_special_tokens=False)
    triples = parse_predictions(pre_triples, )

    new_dict['text'] = sample
    new_dict['spo_list'] = triples
    return new_dict



def query_test(sample, config):
    print('do testing ...')
    model = config.model.to(config.device)

    best_model_path = config.trained_model_path + 'best_f1.pth'
    last_model_path = config.trained_model_path + 'last_model.pth'
    print(f"best_model_path: {best_model_path}")
    print(f"last_model_path: {last_model_path}")

    # Check if best_f1.pth exists, otherwise load last_model.pth
    if os.path.isfile(best_model_path):
        # print   (f"Loading best model from {best_model_path}")
        logger.info(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=config.device))
    elif os.path.isfile(last_model_path):
        # print  (f"Loading last model from {last_model_path}")
        logger.info(f"Loading last model from {last_model_path}")
        model.load_state_dict(torch.load(last_model_path, map_location=config.device))
    else:
        raise FileNotFoundError(f"Neither {best_model_path} nor {last_model_path} exists.")
    # last_model_path = config.trained_model_path + 'best_f1.pth'

    # 模型训练
    result = model2predict(model, config, sample)
    logger.info('------------------test result------------------')
    logger.info(result)

    # 结果输出

    print(f"text : {result['text']}")
    print(f"triples : {result['spo_list']}")
    return result


def fun(text):

    args = get_args()
    # config
    config = Config(args)
    return query_test(text, config)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    question = None
    if request.method == 'POST':
        question = request.form['text']
        result = fun(question)
    return render_template('index.html', question=question, result=result)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
# "I think we could have stood up stronger to the mayor and not bowed to this annual budget dance process that we go through , '' Councilman Charles Barron of Brooklyn said ."
# python3 online_query.py --dataset=nyt  --data_base_dir=./data   --model_name_or_path=../t5-base   --output_dir=./save_model/  --model_type t5   --epochs 20  --batch_size 8 --eval_batch_size 10  --learning_rate 1e-4
