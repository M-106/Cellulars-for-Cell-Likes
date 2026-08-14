
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
- NCA + Latent-Space (global information) -> via MLP add (FiLM = Feature-wise linear modulations)
- NCA + Latent-Space but with Timestep as additional Input for MLP (currently)
- NCA + Global Cross-Attention from Input-Image
    - Only with downsampled balanced classes possible

<br><br>

To Do:
- ResNet with Class Head also as Baseline for comparison
- Test all others with also with aug + test with downsampled balanced classes
    - NCA
    - NCA + FiLM
    - AlexNet
    - AlexNet + NCA
    - AutoEncoder
    - AutoEncoder + NCA
- Test NCA + Global Attention with Latent Space (before tested with Input Image Global Attn)

<br><br>

Ideas:
- NCA + Input-Skip-Connection (?)
- NCA + Global State as additional Input?

<br><br>

> Researchquestion: Does it make sense to use NCAs applying on Feature Space or is it more effective to use them directly or on latent space.

<br><br>

---
### Experiments & Results

> First run, make a "dry run" & a overfit-test. Maybe you have to remove the last activation layer.

Checking the results do:
1. Open Anaconda Prompt
2. Start env and tensorboard
    ```bash
    conda activate cfc && tensorboard --logdir="C:\Users\Shadow\src\Cellulars-for-Cell-Likes\output\2026-08-03_20-10-17_NCA_experiment\logs"
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

<!--

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

-->


| Architecture | Balanced Accuracy | Precision | Recall |  F1 | Addition  | Balanced Class | Augmentation |
|---|---|---|---|---|---|---|---|
| NCA  | 0.2446 | 0.58 | 0.36 | 0.44 | - hidden_channels: 16<br>- steps: 64<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2 | False | False |
| NCA | 0.2850 | 0.57 | 0.35 | 0.4 | - hidden_channels: 16<br>- steps: 64<br>- update_blocks: 2<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2 |  False | False |
| NCA | 0.2944 | 0.68 | 0.43 | 0.5 | - hidden_channels: 16<br>- steps: 8<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2 |  False | False |
| NCA | 0.3258 | 0.55 | 0.36 | 0.42 | - hidden_channels: 8<br>- steps: 64<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2 |  False | False |
| NCA | 0.3502 | 0.66 | 0.41 | 0.49 | - hidden_channels: 8<br>- steps: 8<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2 |  False | False |
| NCA | 0.2854 | 0.59 | 0.36 | 0.44 | - hidden_channels: 4<br>- steps: 8<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2 | False | False |
| NCA | 0.3814 | 0.66 | 0.42 | 0.5 | - hidden_channels: 8<br>- steps: 4<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2 | False | False |
| NCA | 0.4180 | 0.7 | 0.38 | 0.43 | - hidden_channels: 8<br>- steps: 64<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2 | False | False | 
| NCA | 0.3297 | 0.69 | 0.41 | 0.48 | - hidden_channels: 16<br>- steps: 16<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2<br>Hint: Changed something small internally with the dropout. | False | False | <!--2026-07-20_19-07-14_experiment_run_fixed-->
| NCA | 0.3842 | 0.67 | 0.4 | 0.48 | - hidden_channels: 16<br>- steps: 16<br>- update_blocks: 1<br>- final_update_block_activation: "leaky_relu"<br>perception_filter: "sobel"<br>- dropout: 0.2 | False | False | <!-- 2026-07-21_13-31-48_experiment_run_fixed -->
| NCA | 0.2889 | 0.67 | 0.36 | 0.44 | - hidden_channels: 16<br>- steps: 16<br>- update_blocks: 1<br>- final_update_block_activation: "tanh"<br>perception_filter: "sobel"<br>- dropout: 0.2 | False | False | <!-- 2026-07-22_09-21-39_experiment_run_fixed -->
| NCA | 0.2806 | 0.65 | 0.4 | 0.48 | - hidden_channels: 16<br>- steps: 16<br>- update_blocks: 1<br>- final_update_block_activation: "leaky_relu"<br>perception_filter: "learnable"<br>- dropout: 0.2 | False | False | <!-- 2026-07-23_00-55-50_experiment_run_fixed -->
| NCA | 0.4035 | 0.68 | 0.4 | 0.49 | - hidden_channels: 16<br>- steps: 16<br>- update_blocks: 1<br>- final_update_block_activation: "leaky_relu"<br>perception_filter: "pretrained"<br>- dropout: 0.2 | False | False | <!-- 2026-07-23_16-09-31_experiment_run_fixed -->
| NCA | 0.2903 | 0.61 | 0.37 | 0.46 | - hidden_channels: 16<br>- steps: 16<br>- update_blocks: 1<br>- final_update_block_activation: "leaky_relu"<br>perception_filter: "learnable"<br>- dropout: 0.5 | False | False |
| Random | 0.1114 | 0.12 | 0.11 | 0.09 |  | False | False |
| Conv AutoEncoder | 0.2734 | 0.56 | 0.42 | 0.47 | - epochs: 100 (50 latent, 50 class head)<br>- latent_dim: 1014<br>- vae_using_nca: false<br>- lambda_diversity_loss: 0.01 | False | False |  <!--2026-07-28_10-16-56_AE_improved_upsampling_diversity_l1_loss-->
| Conv AutoEncoder | 0.3068 | 0.57 | 0.42 | 0.47 | - epochs: 100 (50 latent, 50 class head)<br>- latent_dim: 1014<br>- vae_using_nca: false<br>- lambda_diversity_loss: 1.5 | False | False |   <!--2026-07-29_09-00-40_AE_improved_upsampling_diversity_l1_loss-->
| Conv AutoEncoder with NCA | 0.1093 | 0.18 | 0.13 | 0.12 | - epochs: 100 (50 latent, 50 class head)<br>- latent_dim: 1014<br>- vae_using_nca: true<br>- lambda_diversity_loss: 1.5 <br>Using NCA as Feature-Refinement after backbone.<br>- hidden_channels: encoder-out-channels<br>- steps: 8<br>- update_blocks: 1<br>- final_update_block_activation: "tanh"<br>perception_filter: "learnable"<br>- dropout: 0.1 | False | False |  <!--2026-07-30_12-44-57_AE_improved_upsampling_diversity_l1_loss_NCA-->
| AlexNet | 0.3034 | 0.59 | 0.41 | 0.47 | - epochs: 100<br>- lr: 0.0005 | False | False |   <!--2026-07-31_18-34-16_AlexNet_run-->
| AlexNet with NCA | 0.2564 | 0.53 | 0.39 | 0.45 |  NCA with AlexNet as Backbone. 75 epochs, because first trained backbone before training NCA head. | False | False |   <!--2026-08-01_13-13-14_AlexNet_run_NCA-->
| NCA | 0.1714 | 0.68 | 0.40 | 0.48 | -epochs: 50<br>- hidden_channels: 16<br>- steps: 16<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "learnable"<br>- dropout: 0.2<br>- lr: 0.0005 (from 0.001)<br>- With Latent Space Update Enhancement (FiLM) | False | False |   <!--2026-08-02_14-56-57_NCA_with_FiLM-->
| NCA | 0.3570 | 0.65 | 0.36 | 0.45 | -epochs: 50<br>- hidden_channels: 16<br>- steps: 8<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "learnable"<br>- dropout: 0.2<br>- lr: 0.0005 (from 0.001)<br>- With Latent Space Update Enhancement (FiLM) | False | False |   <!--2026-08-03_20-10-17_NCA_experiment-->
| NCA | 0.3939 | 0.68 | 0.33 | 0.43 | -epochs: 50<br>- hidden_channels: 16<br>- steps: 8<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "learnable"<br>- dropout: 0.2<br>- With Latent Space Update Enhancement (FiLM) | False | False |   <!--2026-08-04_07-56-30_NCA_experiment_greater_loss-->
| NCA | 0.3938 | 0.69 | 0.42 | 0.49 | -epochs: 50<br>- hidden_channels: 16<br>- steps: 8<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2<br>- With Latent Space Update Enhancement (FiLM) | False | False |   <!--2026-08-04_18-54-47_NCA_experiment-->
| NCA | 0.2893 | 0.65 | 0.41 | 0.48 | -epochs: 50<br>- hidden_channels: 64<br>- steps: 8<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "learnable"<br>- dropout: 0.2<br>- With Latent Space Update Enhancement (FiLM) | False | False | <!--2026-08-05_08-33-45_NCA_experiment-->
| NCA | 0.2685 | 0.62 | 0.41 | 0.48 | -epochs: 100<br>- hidden_channels: 64<br>- steps: 8<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "learnable"<br>- dropout: 0.2<br>- With Latent Space Update Enhancement (FiLM) | False | False | <!--2026-08-06_14-02-19_NCA_experiment-->

<!--
| NCA  |  | - hidden_channels: 16<br>- steps: 64<br>- update_blocks: 1<br>- final_update_block_activation: "sigmoid"<br>perception_filter: "sobel"<br>- dropout: 0.2 |
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

> get right NCA architecture via git!!!
-->

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








