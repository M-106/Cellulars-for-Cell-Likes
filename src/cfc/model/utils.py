# -----------
# > Imports <
# -----------
import torch



# ----------
# > Helper <
# ----------
def init_weights(model):
    if isinstance(model, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            torch.nn.init.constant_(model.bias, 0)












            