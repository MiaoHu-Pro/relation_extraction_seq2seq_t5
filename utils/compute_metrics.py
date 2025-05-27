import json
from sklearn.metrics import precision_recall_fscore_support

# Function to flatten nested lists and dictionaries
def flatten_triples(nested_triples):
    flat_list = []
    for inner_list in nested_triples:
        flat_list.extend(inner_list)
    return flat_list

def compute_metrics(pred_triples, true_triples):
    """
    Compute precision, recall, and F1 score for the given true and predicted triples.

    Args:
        true_triples (list): List of true triples in the format [{'subject': str, 'predicate': str, 'object': str}, ...].
        pred_triples (list): List of predicted triples in the same format.

    Returns:
        dict: A dictionary containing precision, recall, and F1 score.
    """

    print("Computing metrics for triples...")
    # print("true_triples:", true_triples)
    # print("pred_triples:", pred_triples)

    true_set = set((t['subject'], t['predicate'], t['object']) for t in flatten_triples(true_triples))
    pred_set = set((t['subject'], t['predicate'], t['object']) for t in flatten_triples(pred_triples))

    # Calculate precision, recall, and F1 score
    TP = len(true_set & pred_set)
    FP = len(pred_set - true_set)
    FN = len(true_set - pred_set)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

# # Function to parse JSON triples from string
# def parse_triples(triple_str):
#     try:
#         return json.loads(triple_str.replace("'", '"'))
#     except json.JSONDecodeError:
#         return []
#
# if __name__ == "__main__":
#
#     true_labels = []
#     pred_labels = []
#
#     with open('../test_results/t5-base_epochs_50_bs_16_lr_1e-05_test_gen_log.txt', 'r', encoding='utf-8') as file:
#         for line in file:
#             parts = line.strip().split(" ## ")
#             if len(parts) == 3:
#                 true_triples = parse_triples(parts[1])
#                 pred_triples = parse_triples(parts[2])
#                 true_labels.append(set((t['subject'], t['predicate'], t['object']) for t in true_triples))
#                 pred_labels.append(set((t['subject'], t['predicate'], t['object']) for t in pred_triples))
#
#     # Calculate metrics
#     TP = sum(len(true & pred) for true, pred in zip(true_labels, pred_labels))
#     FP = sum(len(pred - true) for true, pred in zip(true_labels, pred_labels))
#     FN = sum(len(true - pred) for true, pred in zip(true_labels, pred_labels))
#
#     precision = TP / (TP + FP) if TP + FP else 0
#     recall = TP / (TP + FN) if TP + FN else 0
#     f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0
#
#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1 Score:", f1_score)
