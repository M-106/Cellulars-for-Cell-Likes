# -----------
# > Imports <
# -----------
import torch



# ---------------------
# > Optimizer Loading <
# ---------------------
def get_optimizer(optimizer_name, model_parameters, learning_rate, weight_decay):
    """
    Load an optimizer based on the provided optimizer name.

    Args:
        optimizer_name (str): Name of the optimizer to load.
        model_parameters (iterable): Parameters of the model to optimize.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.

    Returns:
        torch.optim.Optimizer: The loaded optimizer.
    """
    if optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")






