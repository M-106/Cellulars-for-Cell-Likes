# -----------
# > Imports <
# -----------
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split




# -----------
# > Dataset <
# -----------
class ISIC2019Dataset(Dataset):
    def __init__(self, data_path, partition="val"):
        if partition in ["train", "val"]:
            data_file = f"{data_path}/ISIC_2019_Training_Input/"

            # make split




# ----------
# > Getter <
# ----------
def get_data(
        dataset_name=dataset_name, 
        batch_size=batch_size, 
        partition="val",
        shuffle=False
    ):
    pass

    # load gt

    # load data













