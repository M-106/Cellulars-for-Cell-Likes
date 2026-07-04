# -----------
# > Imports <
# -----------
import torch



# ---------------------
# > Scheduler Loading <
# ---------------------
def get_scheduler(scheduler_name, optimizer, num_epochs):
    """
    Load a scheduler based on the provided scheduler name.

    Args:
        scheduler_name (str): Name of the scheduler to load.
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        num_epochs (int): Total number of epochs for training.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The loaded scheduler.
    """
    if scheduler_name.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported.")







