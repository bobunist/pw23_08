import numpy as np

y_true = ["Cat"] * 6 + ["Fish"] * 9 + ["Hen"] * 10
y_pred = ["Cat"] * 4 + ["Hen"] + ["Fish"] * 6 + ["Cat"] * 2 + ["Hen"] * 6

unique_classes = list(set(y_true))


def calculate_metrics(y_true, y_pred, target_class):
    true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == target_class)
    false_positives = sum(1 for true, pred in zip(y_true, y_pred) if pred == target_class and true != target_class)
    false_negatives = sum(1 for true, pred in zip(y_true, y_pred) if true == target_class and pred != target_class)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score, true_positives


print("\tPrecision\tRecall\tF1-Score\tSupport")
for cls in unique_classes:
    precision, recall, f1_score, support = calculate_metrics(y_true, y_pred, cls)
    print(f"{cls}\t{precision:.2f}\t\t{recall:.2f}\t{f1_score:.2f}\t\t{support}")

total_correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
accuracy = total_correct / len(y_true)
print("\nAccuracy:", accuracy)

precision_avg = np.mean([calculate_metrics(y_true, y_pred, cls)[0] for cls in unique_classes])
recall_avg = np.mean([calculate_metrics(y_true, y_pred, cls)[1] for cls in unique_classes])
f1_avg = np.mean([calculate_metrics(y_true, y_pred, cls)[2] for cls in unique_classes])

support_dict = {cls: y_true.count(cls) for cls in unique_classes}
weighted_precision = sum(precision * support_dict[cls] for cls, precision in zip(unique_classes, [
    calculate_metrics(y_true, y_pred, cls)[0] for cls in unique_classes])) / len(y_true)
weighted_recall = sum(recall * support_dict[cls] for cls, recall in
                      zip(unique_classes, [calculate_metrics(y_true, y_pred, cls)[1] for cls in unique_classes])) / len(
    y_true)
weighted_f1 = sum(f1_score * support_dict[cls] for cls, f1_score in
                  zip(unique_classes, [calculate_metrics(y_true, y_pred, cls)[2] for cls in unique_classes])) / len(
    y_true)

print("Macro Avg\t", f"{precision_avg:.2f}\t\t{recall_avg:.2f}\t{f1_avg:.2f}\t\t{len(y_true)}")
print("Weighted Avg\t", f"{weighted_precision:.2f}\t\t{weighted_recall:.2f}\t{weighted_f1:.2f}\t\t{len(y_true)}")
