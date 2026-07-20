
# Cellulars for Cell-Likes

NCA for Skin Cancer Classification and Segmentation. Cellular logic on cell-like structures. A comparison between SOTA approaches and neural cellular automates.

<br>

> **NCAs on ISIC 2019 Image Only Lesion Diagnosis**


<br><br>

---
### Core Idea

Applying neural cellular automates onto skin cancer image data for classification (and maybe segmentation) and comparing the results to [ISIC 2019 Lesion Diagnosis (Image Only)](https://challenge.isic-archive.com/leaderboards/2019/).




<br><br>

---
### Installation

1. Prepare Env
    1. Install Anaconda
    2. Open Anaconda Prompt
    3. Create env (you might to install another pytorch version):
        ```bash
        conda create -n cfc python=3.14 pip -y
        conda activate cfc
        # pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
        # pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126 --trusted-host download.pytorch.org
        # pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
        pip install -e .
        ```
    4. Quick check:
        ```python
        python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
        ```
2. Download Dataset via https://challenge.isic-archive.com/data/#2019
    - Download `Training Data` Task 1 -> 9,1 GB
    - Download `Training Ground Truth` -> 1 MB
    - Download `Test Data` Task 1 -> 3,6 GB
    - Download `Test Ground Truth` -> 454 KB
    - Unzip everything into one folder
3. Adjust the config file
4. Run it
    1. Activate the env in VS Code (as current used interpreter)
    2. Open any python file + Click on the arrow next to the Run button and choose "Debug using launch.json" and choose cfc then


> Shadow PC PyTorch Block Workaround:
> - comment out all 3 torch related line sin the projecttoml file
> - Use anaconda isntallation instead ```conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia```

<br><br>

---
### Experiments & Results

> First run, make a "dry run" & a overfit-test. Maybe you have to remove the last activation layer.

Checking the results do:
1. Open Anaconda Prompt
2. Start env and tensorboard
    ```bash
    conda activate cfc && tensorboard --logdir="C:\Users\Shadow\src\Cellulars-for-Cell-Likes\output\2026-07-08_22-07-08_experiment_run\logs"
    ```

<!--
"F:\Studium\Master\3.Semester\Teamwork\Cellulars-for-Cell-Likes\output\2026-07-05_14-33-30_overfit\logs"
"C:\Users\Shadow\src\Cellulars-for-Cell-Likes\output\2026-07-08_22-07-08_experiment_run\logs"
"C:\Users\Shadow\src\Cellulars-for-Cell-Likes\output\2026-07-11_18-10-08_experiment_run\logs"
-->

<br>

**Experiment 1: Overfitting on 5 Samples**

Configs:
```yaml
train:
    num_epochs: 200
    batch_size: 5
    learning_rate: 0.001
    weight_decay: 0.0001
    criterion: "focal_loss"
    optimizer: "Adamw"
    scheduler: "cosine"
    output_dir: "./output"
    exp_name: "overfit"
    used_train_samples: 5
    used_val_samples: 5
```

Result on Test-Data with official weighting:
```text
Test Metrics:
  - balanced_accuracy: 0.21658527520156512
  - detailed_report:
              precision    recall  f1-score   support

         MEL       0.33      0.24      0.28      1775
          NV       0.86      0.35      0.50      6137
         BCC       0.00      0.00      0.00         0
          AK       0.02      0.03      0.03       326
         BKL       0.00      0.00      0.00         0
          DF       0.00      0.00      0.00         0
        VASC       0.00      0.00      0.00         0
         SCC       0.00      0.00      0.00         0
         UNK       0.00      0.00      0.00         0

    accuracy                           0.32      8238
   macro avg       0.13      0.07      0.09      8238
weighted avg       0.71      0.32      0.43      8238
```



<br><br>

---
### Approach Idea: Image-Classification-Task NCA

FIXME


<br><br>

---
### Approach Idea: Multi-Task NCA

The NCA grids have 16 channels and the input image come into the first 3 channels. Now the automat runs 32 steps.

Define loss in a way that after 32 steps:
- channel 4 have segmentation mask (loss against gt mask) => IoU Loss
- channel 5 gets one value over the whole widthxheight which is the classification (loss against classification label) -> depends on the classes (just 2?) => Cross-Entropy

Maybe in that way the NCA looks at the shape and symmetry for channel 5 (classification) which would also been done by a doctor (I guess). 


<br><br>

---
### Sources

- Benchmark (Data & Comparison)
    - [ISIC-2019](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification)
    - [Leaderboard of ISIC-2019](https://challenge.isic-archive.com/leaderboards/2019/)
- Dataset (depends on which using in the end: `HAM10000, ISIC-2018, ISIC-2019, and/or ISBI-2020`)
    - [Data-Loader: Fed-ISIC-2019](https://huggingface.co/datasets/flwrlabs/fed-isic2019)
    - [ISIC 2019 Skin Lesion Image Classification](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification)
    - [Backup Dataset: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Cellular Automates:
    - NCAs are better than UNet in Segmentation:
        - `Med-NCA: Robust and Lightweight Segmentation with Neural Cellular Automata` (arxiv, 2023)
        - `NCAdapt` (WACV 2025) \*follow up from 2023
    - NCAs can make good classifications and robust agaisnt domain shift
        - `Neural Cellular Automata for Lightweight, Robust and Explainable Classification of White Blood Cell Images` (MICCAI 2024)
        - collective voting from NCA, shown mathematically: `Self-classifying MNIST Digits` (Mordvintsev et al., 2020)
    - Possible Code basis: [M3D-NCA](https://github.com/MECLabTUDA/M3D-NCA)
    - FIXME (best lib for that? Own?)
- Paper which make similiar things:
    - [Measuring Prediction Uncertainty in Neural Cellular Automata](https://arxiv.org/abs/2605.26726)
    - [Skin cancer segmentation and recognition from dermoscopy images: a novel framework based on improved DeepLabV3+ and network-level fused deep architectures](https://www.sciencedirect.com/science/article/pii/S209012322500654X)




<br><br>

---
### Code

- [M3D-NCA](https://github.com/MECLabTUDA/M3D-NCA)

Or by your self (tempalte code from Gemini):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MedicalNCA(nn.Module):
    def __init__(self, channels=16, hidden_channels=32):
        super().__init__()
        self.channels = channels
        
        # 1. Perception: Jedes Pixel "sieht" seine Nachbarn über Sobel-Filter (Stichwort: lokale Ableitung)
        # In der Praxis nutzt man oft einfach eine Depthwise-Convolution
        self.perceive = nn.Conv2d(channels, channels * 3, kernel_size=3, padding=1, groups=channels, bias=False)
        
        # 2. Update-Regel (Das eigentliche Gehirn der Zelle als 1x1 Convolutions / MLP)
        self.update_net = nn.Sequential(
            nn.Conv2d(channels * 3, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        )
        
        # Gewichte gegen Null initialisieren, damit das Wachstum stabil startet
        nn.init.zeros_(self.update_net[-1].weight)

    def forward(self, x, steps=32, fire_rate=0.5):
        for _ in range(steps):
            # Lokale Wahrnehmung der Nachbarpixel
            perception = self.perceive(x)
            
            # Update berechnen
            ds = self.update_net(perception)
            
            # Stochastisches Update (Zellen updaten sich biologisch-zufällig, nicht alle synchron)
            rand_mask = (torch.rand(x.size(0), 1, x.size(2), x.size(3), device=x.device) < fire_rate).float()
            
            # Zustand des Gitters aktualisieren
            x = x + ds * rand_mask
            
        return x
```


> See your costum train pipeline in MCR-Lab as good starting point for the pipeline.





