import json
from sklearn.metrics import precision_recall_fscore_support

# Function to parse JSON triples from string
def parse_triples(triple_str):
    try:
        return json.loads(triple_str.replace("'", '"'))
    except json.JSONDecodeError:
        return []

true_labels = []
pred_labels = []

with open('../test_results/t5-large_epochs_50_bs_16_lr_0.0001_test_gen_log.txt', 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split(" ## ")
        if len(parts) == 3:
            true_triples = parse_triples(parts[1])
            pred_triples = parse_triples(parts[2])
            true_labels.append(set((t['subject'], t['predicate'], t['object']) for t in true_triples))
            pred_labels.append(set((t['subject'], t['predicate'], t['object']) for t in pred_triples))

# Calculate metrics
TP = sum(len(true & pred) for true, pred in zip(true_labels, pred_labels))
FP = sum(len(pred - true) for true, pred in zip(true_labels, pred_labels))
FN = sum(len(true - pred) for true, pred in zip(true_labels, pred_labels))

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
