# -----------
# > Imports <
# -----------
import torch



# ---------------------
# > Scheduler Loading <
# ---------------------
def get_scheduler(scheduler_name, optimizer, num_epochs, warmup_epochs=2):
    """
    Load a scheduler based on the provided scheduler name.

    Args:
        scheduler_name (str): Name of the scheduler to load.
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        num_epochs (int): Total number of epochs for training.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The loaded scheduler.
    """
    warumup_steps = warmup_epochs # * epoch_iters
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,  # 10% of LR
        end_factor=1.0,
        total_iters=warumup_steps
    )

    if scheduler_name.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name.lower() == "step":
        scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported.")

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, scheduler],
        milestones=[warumup_steps]
    )







