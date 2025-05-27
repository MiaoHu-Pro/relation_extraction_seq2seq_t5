import argparse
import re

import torch
import json
import pathlib


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_base_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--output_dir",
                        default='./save_model/',
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="which model to use, bert, roberta, gpt2, llama, t5, mt5, bart")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list:")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="dataset name")

    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=400,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--hidden_dim",
                        default=768,
                        type=int,
                        help="the dim of hidden victor.")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do_develop",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")

    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--epochs",
                        default=10,
                        type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument('--step',
                        type=int,
                        default=1,
                        help="random seed for initialization")
    parser.add_argument('--max_len',
                        type=int,
                        default=512,
                        help="max length of inout text")
    parser.add_argument('--data_parallel', default=False, help='using multiple gpus or not')

    parser.add_argument('--max_len_generate', default=100, help='the max length of generated text')


    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                        help="")
    return   parser.parse_args()

def print_and_save_params(config, filepath):
    exclude_keys = {'tokenizer', 'model', 'id2rel', 'rel_vocab'}
    params = {str(key): (vars(value) if hasattr(value, '__dict__') else value) for key, value in config.__dict__.items() if key not in exclude_keys}


    # Convert non-serializable objects to strings
    for key, value in params.items():
        if isinstance(value, torch.device):
            params[key] = str(value)
        elif isinstance(value, pathlib.Path):
            params[key] = str(value)
        elif isinstance(value, set):
            params[key] = list(value)
        elif callable(value):
            params[key] = str(value)
        elif isinstance(value, object) and not isinstance(value, (str, int, float, list, dict)):
            params[key] = str(value)

    # Save parameters to file
    try:
        with open(filepath + 'parameters.json', 'w') as f:
            json.dump(params, f, indent=4)
        print(f"Parameters saved to {filepath}")
    except TypeError as e:
        print(f"Error saving parameters: {e}")
def parse_predictions_1(preds):
    # print('parse_predictions', preds)

    """
    pred_str: a single string which may contain one or more triples
              separated by '<TRIPLE>'.
    returns:  list of dicts with keys 'head', 'relation', 'tail'
    """

    # pred_str = preds[0]

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
    batch_triples = []
    for pred_str in preds:

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
                "subject": head,
                "object": tail,
                "predicate": relation,
            })

        batch_triples.append(triples)

    return batch_triples


def parse_predictions(preds):
    pattern = re.compile(
        r"""(?x)                # verbose regex
        (?:<SUBJ>\s*)?          # optional <SUBJ> tag
        (?P<head>\S+)           # the head is the next non‐space run
        \s+<PRED>\s+            # the <PRED> tag
        (?P<relation>\S+)       # the relation (one token)
        \s+<OBJ>\s+             # the <OBJ> tag
        (?P<tail>.+)$           # the rest of the segment is the tail
        """
    )
    batch_triples = []
    for pred_str in preds:
        segments = (
            pred_str
                .replace("<pad>", "")
                .replace("<s>", "")
                .replace("</s>", "")
                .split("<TRIPLE>")
        )
        print("\n\nsegments: ", segments)
        triples = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            m = pattern.match(seg)
            if not m:
                continue
            head = m.group("head")
            relation = m.group("relation")
            tail = m.group("tail").strip()
            triples.append({
                "subject": head,
                "object": tail,
                "predicate": relation,
            })
        batch_triples.append(triples)
    return batch_triples

if __name__ == "__main__":
    str_list = [     '<pad>Annan <PRED> /people/person/nationality <OBJ> Iran</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>',
        '<pad>McFarlane <PRED> /business/person/company <OBJ> Toys</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>',
        '<pad>Damascus <PRED> /location/administrative_division/country <OBJ> Syria <TRIPLE> <SUBJ> Syria <PRED> /location/country/capital <OBJ> Damascus <TRIPLE> <SUBJ> Syria <PRED> /location/location/contains <OBJ> Damascus <TRIPLE> <SUBJ> Syria <PRED> /location/country/administrative_divisions <OBJ> Damascus</s>',
        '<pad>Afghanistan <PRED> /location/country/capital <OBJ> Kabul <TRIPLE> <SUBJ> Afghanistan <PRED> /location/location/contains <OBJ> Kabul</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>',
        '<pad>Hills <PRED> /location/neighborhood/neighborhood_of <OBJ> Angeles <TRIPLE> <SUBJ> Angeles <PRED> /location/location/contains <OBJ> Hills</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>',
        '<pad>York <PRED> /location/location/contains <OBJ> Bridge</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>',
        '<pad>Israel <PRED> /location/location/contains <OBJ> Bank <TRIPLE> <SUBJ> Israel <PRED> /location/country/capital <OBJ> Jerusalem</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>',
        '<pad>Myanmar <PRED> /location/country/capital <OBJ> Mandalay <TRIPLE> <SUBJ> Myanmar <PRED> /location/location/contains <OBJ> Mandalay</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>',
        '<pad>Lebanon <PRED> /location/country/capital <OBJ> Beirut <TRIPLE> <SUBJ> Lebanon <PRED> /location/location/contains <OBJ> Beirut</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>',
        '<pad>Annan <PRED> /business/person/company <OBJ> Nations</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>',
        '<pad>Israel <PRED> /location/location/contains <OBJ> Bank</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>',
        '<pad>Missouri <PRED> /location/location/contains <OBJ> Louis</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>',
        '<pad>Afghanistan <PRED> /location/location/contains <OBJ> Herat</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>',
        '<pad>Flatbush <PRED> /location/neighborhood/neighborhood_of <OBJ> Brooklyn <TRIPLE> <SUBJ> Brooklyn <PRED> /location/location/contains <OBJ> Flatbush</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>',
        '<pad>Ahmadinejad <PRED> /people/person/nationality <OBJ> Iran</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>',
        '<pad>Israel <PRED> /location/location/contains <OBJ> Bank</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>']

    result = parse_predictions(str_list)
    print("Result:", result)
