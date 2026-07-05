# -----------
# > Imports <
# -----------
import numpy as np

from sklearn.metrics import balanced_accuracy_score, classification_report



# -----------
# > Metrics <
# -----------
def get_used_label_names(y_true, y_pred, idx_to_class):
    classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    # used_labels = [classes[i] for i in unique_labels]
    used_labels = [idx_to_class[i] for i in unique_labels]
    return used_labels

def calculate_isic_metrics(y_true, y_pred, used_label_names, sample_weights=None):
    """
    y_true: Liste oder Array der tatsächlichen Labels (0-8)
    y_pred: Liste oder Array der vorhergesagten Labels (0-8)
    """
    bacc = balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weights)
    
    report = classification_report(y_true, y_pred, target_names=used_label_names, zero_division=0)  # output_dict=True
    
    return {
        "balanced_accuracy": bacc,
        "detailed_report": report
    }











