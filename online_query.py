# -*- coding: utf-8 -*-
import logging
import re
from test import *
from flask import Flask, render_template, request, jsonify, make_response
from utils.utilities import get_args
from utils.dataset import get_input
from utils.config import *
app = Flask(__name__)

logger = logging.getLogger(__name__)


def parse_predictions(preds):
    print('parse_predictions', preds)

    """
    pred_str: a single string which may contain one or more triples
              separated by '<TRIPLE>'.
    returns:  list of dicts with keys 'head', 'relation', 'tail'
    """

    pred_str = preds[0]
    print('pred_str', pred_str)
    pattern = re.compile(
        r"""(?x)                # verbose regex
        (?:<SUBJ>\s*)?          # optional <SUBJ> tag
        (?P<head>\S+)           # the head is the next non‐space run
        \s+<OBJ>\s+             # the <OBJ> tag
        (?P<relation>\S+)       # the relation (one token)
        \s+<PRED>\s+            # the <PRED> tag
        (?P<tail>.+)$           # the rest of the segment is the tail
        """
    )

    # 1) strip off any global <pad> or </s>, split on <TRIPLE>
    segments = (
        pred_str
            .replace("<pad>", "")
            .replace("</s>", "")
            .split("<TRIPLE>")
    )
    triples = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        m = pattern.match(seg)
        if not m:
            # you can choose to log or raise here
            continue
        head = m.group("head")
        relation = m.group("relation")
        tail = m.group("tail").strip()
        triples.append({
            "head": head,
            "relation": relation,
            "tail": tail
        })
    return triples

    # parsed = []
    # for s in preds:
    #     parts = s.split(maxsplit=2)
    #     if len(parts) == 3:
    #         head, relation, tail = parts
    #         parsed.append({
    #             "head": head,
    #             "relation": relation,
    #             "tail": tail
    #         })
    #
    # return parsed

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
