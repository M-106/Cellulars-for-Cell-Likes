# -----------
# > Imports <
# -----------
import torch
import torch.nn as nn



# ---------
# > Model <
# ---------

class RandomClassifier(nn.Module):
    def __init__(self,
        num_classes=10,  
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        


    def forward(self, x):
        return torch.rand(size=(x.shape[0], self.num_classes), device=x.device)














