
# Cellulars for Cell-Likes

NCA for Skin Cancer Classification and Segmentation. Cellular logic on cell-like structures. A comparison between SOTA approaches and neural cellular automates.

<br>

> **NCAs on ISIC 2019 Image Only Lesion Diagnosis**

<br><br>

- [Core Idea](#core-idea)
- [Installation](#installation)
- [Experiment Plan](#experimentplan)
- [Experiments & Results](#experiments--results)
- [Sources](#sources)


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
### Experimentplan

- Few-Sample-Overfitting
- "Standard" Training
- Vary Hidden Channel Amount
- Vary Steps
- Vary with Perception (sobel, trained, pretrained)
- AlexNet Inspired Net
- AlexNet Inspired + NCA (on Backbone Feature Space)
- AutoEncoder + NCA
- NCA + Latent-Space (global information) -> via MLP add
- NCA + Input-Skip-Connection (?)
- NCA + Global State as additional Input?

> Researchquestion: Does it make sense to use NCAs applying on Feature Space or is it more effective to use them directly or on latent space.

<br><br>

---
### Experiments & Results

> First run, make a "dry run" & a overfit-test. Maybe you have to remove the last activation layer.

Checking the results do:
1. Open Anaconda Prompt
2. Start env and tensorboard
    ```bash
    conda activate cfc && tensorboard --logdir="C:\Users\Shadow\src\Cellulars-for-Cell-Likes\output\2026-07-28_02-07-07_AE_first_try\logs"
    ```

<!--
"F:\Studium\Master\3.Semester\Teamwork\Cellulars-for-Cell-Likes\output\2026-07-05_14-33-30_overfit\logs"
"C:\Users\Shadow\src\Cellulars-for-Cell-Likes\output\2026-07-08_22-07-08_experiment_run\logs"
"C:\Users\Shadow\src\Cellulars-for-Cell-Likes\output\2026-07-11_18-10-08_experiment_run\logs"
"C:\Users\Shadow\src\Cellulars-for-Cell-Likes\output\2026-07-20_19-07-14_experiment_run_fixed\logs"
"C:\Users\Shadow\src\Cellulars-for-Cell-Likes\output\2026-07-21_13-31-48_experiment_run_fixed\logs"
"2026-07-23_16-09-31_experiment_run_fixed"
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


**Overall Results:**

Standard Values are:
* Epochs: 50
* Learning-Rate: 1e-3
* Optimizer: Adam-W

> Most models already reached a convergence/stagnation at 50 epochs.


| Architecture | Balanced Accuracy | Addition | Converged |
|---|---|---|---|
| NCA  | 20.84 | - hidden_channels: 16<br>- steps: 64<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2 | True |
| NCA | 24.23  | - hidden_channels: 16<br>- steps: 64<br>- update_blocks: 2<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2 | True |
| NCA | 28.89 | - hidden_channels: 16<br>- steps: 8<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2 | False |
| NCA | 22.41 | - hidden_channels: 8<br>- steps: 64<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2 | True |
| NCA | 27.14 | - hidden_channels: 8<br>- steps: 8<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2 | True |
| NCA | 20.81 | - hidden_channels: 4<br>- steps: 8<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2 | True |
| NCA | 25.09 | - hidden_channels: 8<br>- steps: 4<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2 | True |
| NCA | 26.48 | - hidden_channels: 8<br>- steps: 64<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2 | False |
| NCA | 27.30 | - hidden_channels: 16<br>- steps: 16<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2<br>Hint: Changed something small internally with the dropout. | False |
| NCA | 23.87 | - hidden_channels: 16<br>- steps: 16<br>- update_blocks: 1<br>- final_update_block_activation: "leaky_relu"<br>perception_filter: "sobel"<br>- dropout: 0.2 | True |
| NCA | 21.71 | - hidden_channels: 16<br>- steps: 16<br>- update_blocks: 1<br>- final_update_block_activation: "tanh"<br>perception_filter: "sobel"<br>- dropout: 0.2 | True |
| NCA | 24.98 | - hidden_channels: 16<br>- steps: 16<br>- update_blocks: 1<br>- final_update_block_activation: "leaky_relu"<br>perception_filter: "learnable"<br>- dropout: 0.2 | False |
| NCA | 24.13 | - hidden_channels: 16<br>- steps: 16<br>- update_blocks: 1<br>- final_update_block_activation: "leaky_relu"<br>perception_filter: "pretrained"<br>- dropout: 0.2 | False |
| NCA | 22.28 | - hidden_channels: 16<br>- steps: 16<br>- update_blocks: 1<br>- final_update_block_activation: "leaky_relu"<br>perception_filter: "learnable"<br>- dropout: 0.5 | True |
| ConvAE | 45.12 | - epochs: 100 (50 latent, 50 class head)<br>- latent_dim: 1014<br>- vae_using_nca: false<br>- lambda_diversity_loss: 1.5 | True |
| ConvAE | 39.36 | - epochs: 100 (50 latent, 50 class head)<br>- latent_dim: 1014<br>- vae_using_nca: true<br>- lambda_diversity_loss: 1.5 <br>Using NCA as Feature-Refinement after backbone.<br>- hidden_channels: encoder-out-channels<br>- steps: 8<br>- update_blocks: 1<br>- final_update_block_activation: "tanh"<br>perception_filter: "learnable"<br>- dropout: 0.1 | True |
| AlexNet | 43.09 |  | True |
| AlexNet-NCA | 48.03 | NCA with AlexNet as Backbone. 75 epochs, because first trained backbone before training NCA head. | True |
| NCA with Latent-FiLM | 24.63 | NCA update-steps multiply and add context from latent-space feed through a MLP. | False |
| NCA |  | - hidden_channels: 16<br>- steps: 8<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "learnable"<br>- dropout: 0.2 | ? |
| NCA |  | - hidden_channels: 64<br>- steps: 8<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "learnable"<br>- dropout: 0.2 | ? |
| NCA |  | -epochs: 100<br>- hidden_channels: 64<br>- steps: 8<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "learnable"<br>- dropout: 0.2 | ? |

> Converged is only if the slope in the last 15 epochs is above 0.001. This still can mean that the model is already converged.



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








