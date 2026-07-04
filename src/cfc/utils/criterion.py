# -----------
# > Imports <
# -----------
import torch
import torch.nn as nn
import torch.nn.functional as F



# ---------------------
# > Define Focal Loss <
# ---------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()



# ---------------------
# > Criterion Loading <
# ---------------------
def get_criterion(criterion_name, class_weights=None):
    """
    Load a criterion (loss function) based on the provided criterion name.

    Reduction="none" is used to return the loss for each sample, allowing for custom aggregation later.

    Args:
        criterion_name (str): Name of the criterion to load.
        class_weights (torch.Tensor, optional): Weights for each class.

    Returns:
        torch.nn.Module: The loaded criterion.
    """
    if criterion_name.lower() == "cross_entropy":
        return torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1, reduction='none')
    
    # Focal Loss ist der Goldstandard bei starkem Ungleichgewicht
    elif criterion_name.lower() == "focal_loss":
        return FocalLoss(gamma=2.0, alpha=class_weights)
    
    else:
        raise ValueError(f"Criterion {criterion_name} not supported.")
    
    # if criterion_name.lower() == "cross_entropy":
    #     return torch.nn.CrossEntropyLoss()
    # elif criterion_name.lower() == "mse":
    #     return torch.nn.MSELoss()
    # else:
    #     raise ValueError(f"Criterion {criterion_name} not supported.")
    









