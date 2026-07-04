# -----------
# > Imports <
# -----------
import numpy as np

from sklearn.metrics import balanced_accuracy_score, classification_report



# -----------
# > Metrics <
# -----------
def calculate_isic_metrics(y_true, y_pred, sample_weights=None):
    """
    y_true: Liste oder Array der tatsächlichen Labels (0-8)
    y_pred: Liste oder Array der vorhergesagten Labels (0-8)
    """
    classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

    bacc = balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weights)
    
    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)  # output_dict=True
    
    return {
        "balanced_accuracy": bacc,
        "detailed_report": report
    }











