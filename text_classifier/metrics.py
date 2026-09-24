"""Classification metrics with a stable label order."""

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(
    truth: list[int], predictions: list[int], labels: list[str]
) -> dict:
    if len(truth) != len(predictions) or not truth:
        raise ValueError("Truth and predictions must have the same nonzero length")
    ids = list(range(len(labels)))
    result = {"accuracy": float(accuracy_score(truth, predictions)), "support": len(truth)}
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predictions, labels=ids, average=average, zero_division=0
        )
        result[average] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=ids, average=None, zero_division=0
    )
    result["per_class"] = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(labels)
    }
    result["labels"] = labels
    result["confusion_matrix"] = confusion_matrix(truth, predictions, labels=ids).tolist()
    return result
