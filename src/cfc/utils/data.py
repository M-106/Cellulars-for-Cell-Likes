# -----------
# > Imports <
# -----------
import os

import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from torchvision.transforms.functional import to_tensor
# from torchvision.io import read_image

from sklearn.model_selection import train_test_split




# -----------
# > Dataset <
# -----------
class ISIC2019Dataset(Dataset):
    def __init__(self, 
                 data_path, 
                 partition="val", 
                 transform=None, 
                 used_samples=-1,
                 balance_classes=False,
                 random_state=42):
        """
        Random split with seed for reproducability. Stratified split to maintain class distribution.

        Labels get preprocessed so that they look like:
        - MEL → 0
        - NV  → 1
        - BCC → 2
        - AK  → 3
        - BKL → 4
        - DF  → 5
        - VASC → 6
        - SCC → 7
        - UNK → 8
        """
        self.transform = transform
        self.class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}

        if partition in ["train", "val"]:
            self.img_folder = f"{data_path}/ISIC_2019_Training_Input/"

            self.labels = pd.read_csv(f"{data_path}/ISIC_2019_Training_GroundTruth.csv")
            self.labels["label"] = (
                self.labels[self.class_names]
                .idxmax(axis=1)
                .map(self.class_to_idx)
            )

            # print(self.labels.head())
            # print("Columns:", self.labels.columns)
            # print("Unique labels in DF:", self.labels["label"].unique())
            # print("Keys in class_to_idx:", list(self.class_to_idx.keys()))

            # make split - reproducable
            train_df, val_df = train_test_split(
                self.labels, 
                test_size=0.2, 
                stratify=self.labels["label"], 
                random_state=random_state
            )

            # if partition == "train":
            #     self.labels = train_df
            # else:
            #     self.labels = val_df

            if partition == "train":
                self.labels = train_df.reset_index(drop=True)
            else:
                self.labels = val_df.reset_index(drop=True)

            # print(self.labels.head())
            # print("Columns:", self.labels.columns)
            # print("Unique labels in DF:", self.labels["label"].unique())
            # print("Keys in class_to_idx:", list(self.class_to_idx.keys()))

        else:
            self.img_folder = f"{data_path}/ISIC_2019_Test_Input/"
            file_path = os.path.join(data_path, "ISIC_2019_Test_GroundTruth.csv")

            # print(f"Exists: {os.path.exists(file_path)}")
            # print(f"Size: {os.path.getsize(file_path)} Bytes")
            
            labels_df = pd.read_csv(file_path)
            labels_df["label"] = (
                labels_df[self.class_names]
                .idxmax(axis=1)
                .map(self.class_to_idx)
            )
            self.labels = labels_df.reset_index(drop=True)

        # Class Balancing / Undersampling Logic
        if balance_classes:
            # Find the sample count of the smallest class
            min_class_count = self.labels["label"].value_counts().min()

            # If used_samples is set, cap the per-class limit accordingly
            if used_samples > 0:
                samples_per_class = min(min_class_count, used_samples // len(self.class_names))
            else:
                samples_per_class = min_class_count

            # Downsample each class to `samples_per_class` entries
            self.labels = (
                self.labels.groupby("label", group_keys=False)
                .sample(n=samples_per_class, random_state=random_state)
                .reset_index(drop=True)
            )
            print(f"Achieve Balanced Class-Samples by downsampling to {samples_per_class} samples per class.")
        elif used_samples > 0:
            # Simple random reduction without balancing
            self.labels = self.labels.sample(n=used_samples, random_state=random_state).reset_index(drop=True)
        else:
            self.labels = self.labels.reset_index(drop=True)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        img_path = f"{self.img_folder}/{img_name}.jpg"
        # image = torch.load(img_path, weights_only=False)
        image = Image.open(img_path).convert("RGB")
        # image = to_tensor(image)
        # image = read_image(img_path)
        # image = image.float() / 255.0

        if self.transform:
            image = self.transform(image)

        col_idx = self.labels.columns.get_loc('label')
        label = self.labels.iloc[idx, col_idx]
        # label = self.labels["label"].iloc[idx]
        # label = self.labels.at[idx, 'label']
        label = torch.tensor(label, dtype=torch.long)
        # label = torch.tensor(self.class_to_idx[label], dtype=torch.long)
        # torch.tensor(self.labels_df.iloc[idx, 1:].values.argmax())

        score_weight = self.labels["score_weight"].iloc[idx] if "score_weight" in self.labels.columns else 1.0
        validation_weight = self.labels["validation_weight"].iloc[idx] if "validation_weight" in self.labels.columns else 1.0

        return image, label, score_weight, validation_weight



# ----------
# > Getter <
# ----------
def get_data(
        data_path, 
        batch_size=16, 
        partition="val",
        shuffle=False,
        used_samples=-1,
        balance_classes=False,
        img_size=(224, 224),
        augmentation=False
    ):

    # Standard-Normalization for Pretrained Models (e.g. ImageNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if partition == "train" and augmentation:
        # Dynamic Augmentation for training
        transformations = transforms.Compose([
            transforms.Resize(img_size),
            
            # --- AUGMENTATIONs ---
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180), # Hautläsionen haben keine feste Ausrichtung
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            # ----------------------
            
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Only Preprocessing for Val/Test without Random-Components
        transformations = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    dataset = ISIC2019Dataset(data_path=data_path, partition=partition, transform=transformations, used_samples=used_samples, balance_classes=balance_classes)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)  # num_workers=0













