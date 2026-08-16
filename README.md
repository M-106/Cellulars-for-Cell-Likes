
# Cellulars for Cell-Likes

NCA for Skin Cancer Classification. Cellular logic on cell-like structures. A comparison between SOTA approaches and neural cellular automates.

<br>

> **NCAs on **

<br><br>

Table of Content
- [Introduction](#introduction)
- [Related Work](#related-work)
    - [DL in Dermoscopic Image Analysis](#dl-in-dermoscopic-image-analysis)
    - [Neural Cellular Automata in Medical Imaging](#neural-cellular-automata-in-medical-imaging)
    - [Limitations of NCAs for Image Classification](#limitations-of-ncas-for-image-classification)
- [Methodology](#methodology)
    - [Data](#data)
    - [Hardware](#hardware)
    - [Software Setup](#software-setup)
    - [Model](#model)
- [Experiments](#experiments)
- [Results](#results)
    - [Internal Experiment Results](#internal-experiment-results)
    - [Placing in the ISIC 2019 Leaderboard](#placing-in-the-isic-2019-leaderboard)
- [Discussion](#discussion)
- [Limitations](#limitations)
- [Conclusion & Future Work](#conclusion--future-work)


<br><br>

---
### Introduction

<!--goal-->
Our project "Cellulars for Cell-Likes" wants to apply neural cellular automates onto skin cancer image data for classification and comparing the results to the leaderboard of [ISIC 2019 Lesion Diagnosis (Image Only)](https://challenge.isic-archive.com/leaderboards/2019/). So our project is fully wrapped around the [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/).<br>
The Challenge provides train and test data from the BCN_20000 Dataset, HAM10000 Dataset, and MSK Dataset. Overall there are 8 categories<!--[[0]](https://doi.org/10.48550/arXiv.1803.10417)-->:
1. Melanoma
2. Melanocytic nevus
3. Basal cell carcinoma
4. Actinic keratosis
5. Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
6. Dermatofibroma
7. Vascular lesion
8. Squamous cell carcinoma
9. None of the others 
The goal metric is a normalized multi-class accuracy metric also called balanced accuracy.<br>
It is a challenging task with images with many varities.

Our motivation for this project is split into 2. First, to our knowledge there is no Neural Cellular Automata Approach tested on this challenge and second, it is known that there is a performance gap between NCAs and large, more complex architectures in image classification [[1]](https://doi.org/10.48550/arXiv.2508.12324)[[2]](https://doi.org/10.48550/arXiv.2404.05584) which is also caused due to the missing global context and too much variety in data, and we want to do different experiments to investigate in closing this gap. 

<br><br>

---
### Related Work

##### DL in Dermoscopic Image Analysis

Deep learning models have established SOTA benchmarks in dermatological image analysis, like the ISIC 2019. Standard approaches heavily rely on CNNs (such as EfficientNets and ResNets) [[3]](https://doi.org/10.1109/tbme.2019.2915839)[[4]](https://doi.org/10.1016/j.dajour.2023.100278)[[5]](https://doi.org/10.3390/bioengineering11080810)[[6]](https://www.researchgate.net/publication/384893629_Benchmarking_Deep_Learning_Models_for_Dermatological_Image_Analysis_EfficientNet_Takes_the_Lead) and Vision Transformers, achieving balanced accuracies up to 89.5%, depending on the benchmark and the time of the benchmark. They often use multi-scale feature extraction, heavy ensemble strategies and external data to reach such a high accuracy. <br>
These models require millions of parameters, high compuational costs, and are difficult to interpret, limiting their diagnostic valuability in clinical environments [[7]](https://doi.org/10.48550/arXiv.2005.02000)[[8]](https://doi.org/10.48550/arXiv.2203.08807).<br>
Hint: During the ISIC 2019, Vision Transformer where not etablished yet, so competitors used CNN-based networks.

<br>

##### Neural Cellular Automata in Medical Imaging

Neural Cellular Automata (NCAs) have emerged in recent years as lightweight, parameter-efficient alternatives that rely only on localized, iterative updates to achieve global state transitions [[9]](https://doi.org/10.23915/distill.00023)[[10]](https://doi.org/10.48550/arXiv.2508.12322)[[11]](https://doi.org/10.1038/s44335-025-00026-4).<br>
Particulary in medical domain applications, NCAs shown substantial success in image segmentation and sythesis tasks. For instance, [MedSegDiffNCA (Mittal et al., 2025)](https://doi.org/10.48550/arXiv.2501.02447) demonstrated that combining NCAs with diffusion models achieves competitive DICE segmentation scores on ISIC datasets while reducing parameter counts by over 60x compared to standard U-net backbones. Similarly, [Yue et al. (2024)](https://doi.org/10.1016/j.bspc.2024.106547) integrated NCAs into UNet latent bottlenecks for skin lesion segmentation, confirming that local cell rules can capture spatial structures effectivly.

<br>

##### Limitations of NCAs for Image Classification

Despite their success in dense spatial tasks (like segmentation), applying pure NCAs to multi-class image classification remains fundamentally challenging [[1]](https://doi.org/10.48550/arXiv.2508.12324)[[2]](https://doi.org/10.48550/arXiv.2404.05584)[[12]](https://doi.org/10.48550/arXiv.1809.02942)[[13]](https://doi.org/10.48550/arXiv.2607.24529). Recent work highlights a persistent performance gap between NCAs and standard vision backbones when summarizing global semantics. [Deutges et al. (2024)](https://doi.org/10.1007/978-3-031-72384-1_65) noted that converting spatial cell grids into a single categorical prediction forces the local update rule to handle both spatial feature propagation and global dimensionality reduction simultaneously, because of that that they tried to split these tasks up.<br>
Furthermore, [Yang et al. (2025)](https://doi.org/10.48550/arXiv.2508.12324) identified that standard local neighborhoods (3x3) struggle to propagate fine-grained diagnostic features across large images without losing context, necessitating specialized pooling layers (such as attention pooling) to bridge the local-to-global aggrgeation bottleneck.


<br><br>

---
### Methodology

##### Data

We used the data from ISIC 2019 Image Only Lesion Diagnosis Challenge with 25.332 training images and 8.239 test samples. During training we used 20% of the train data as validation set.<br>
Our data pipeline includes a normalization step using the ImageNet mean and standard deviation.<br>
Augmentation was optional (specified in the experiment details if used). Our augmentation is a small collection of standard techniques:
* Horizontal Flip
* Vertical Flip
* Rotation (up to 180°)
* Color Jitter (brightness=0.1, contrast=0.1, saturation=0.1)
<br>

We also have an optional downsample to have a balanced amount of each class, also specified if used.

<br>

<a href="https://www.researchgate.net/figure/Samples-of-the-2019-ISIC-dataset-aMelanoma-MEL-bMelanocytic-Nevus-NV-cBasal_fig3_351089784"><img src="https://www.researchgate.net/publication/351089784/figure/fig3/AS:11431281179150051@1691159178365/Samples-of-the-2019-ISIC-dataset-aMelanoma-MEL-bMelanocytic-Nevus-NV-cBasal.png" alt="Samples of the 2019 ISIC dataset. (a) Melanoma – MEL, (b) Melanocytic Nevus – NV, (c) Basal Cell Carcinoma – BCC, (d) Actinic Keratosis – AK, (e) Benign Keratosis – BKL, (f) Dermatofibroma – DF, (g) Vascular Lesion – VASC, (h) Squamous Cell Carcinoma – SCC"/>Samples of the 2019 ISIC dataset. (a) Melanoma – MEL, (b) Melanocytic Nevus – NV, (c) Basal Cell Carcinoma – BCC, (d) Actinic Keratosis – AK, (e) Benign Keratosis – BKL, (f) Dermatofibroma – DF, (g) Vascular Lesion – VASC, (h) Squamous Cell Carcinoma – SCC<br><sub><sup>From: A deep analysis on high‐resolution dermoscopic image classification - Scientific Figure on ResearchGate. Available from: https://www.researchgate.net/figure/Samples-of-the-2019-ISIC-dataset-aMelanoma-MEL-bMelanocytic-Nevus-NV-cBasal_fig3_351089784 [accessed 16 Aug 2026]</sup></sub></a>





<br><br>

##### Hardware

We used *Shadow PC* to harnessing a [NVIDIA RTX 2000 Ada Generation](https://www.nvidia.com/en-us/products/workstations/rtx-2000/) with 16 GB VRAM, NVidia-Driver-Version 565.90 and CUDA-Version 12.7.

<br><br>

##### Software Setup

We used Python 3.14 with torch 2.5.1 and torchvision 0.20.1. A complete list of our dependencies are in the [requirements.txt](./requirements.txt).<br>
The installation can be reproduced following:
1. Prepare Env
    1. Install Anaconda
    2. Open Anaconda Prompt
    3. Create env (you might to install another pytorch version):
        ```bash
        conda create -n cfc python=3.14 pip -y
        conda activate cfc

        # install torch cuda version:
        # pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
        # pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126 --trusted-host download.pytorch.org
        # pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

        # or maybe use the conda version:
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

        # install our repo and the needed dependencies (expect torch and torchvision)
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
3. Adjust the [config file](./configs/config.yaml)
4. Run it
    1. Open a python file and activate the env in VS Code (as current used interpreter)
    2. Click on the arrow next to the Run button and choose "Debug using launch.json" and choose cfc then


<br><br>

##### Model

We built a flexible framework, enabeling different NCA architecture-details directly from the configuration file.<br>
Our base NCA looks like:
```text

```


<br><br>

---
### Experiments

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
### Results

<!--
> First run, make a "dry run" & a overfit-test. Maybe you have to remove the last activation layer.

Checking the results do:
1. Open Anaconda Prompt
2. Start env and tensorboard
    ```bash
    conda activate cfc && tensorboard --logdir="C:\Users\Shadow\src\Cellulars-for-Cell-Likes\output\2026-08-03_20-10-17_NCA_experiment\logs"
    ```
-->

##### Internal Experiment Results

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


##### Placing in the ISIC 2019 Leaderboard



| Team | Approach | Balanced Accuracy |
| --- | --- | --- |
| DAISYLab<br>Hamburg University of Technology/University Medical Center Hamburg-Eppendorf | Ensemble of Multi-Res EfficientNets + SEN154 2 | 0.636 |
| DysionAI<br>DYSION AI, Inc, Beijing, China |  | 0.607 |
| AImageLab & PRHLT<br>Unimore & UPV | ensemble, ood threshold 100% | 0.593 |
| DermaCode | 13 models + hierarchical approach to select outliers | 0.578 |
| Nurithm Labs | Densenet-161 with heavy use of random crops | 0.569 |
| Torus Actions | Simple test approach | 0.563 |
| BITDeeper<br>Beijing Institute of Technology | MelaNet: A Deep Dense Attention Network for Melanoma Detection in Dermoscopy Images | 0.558 |
| SYSU-MIA-Group<br>Sun Yat-sen University Medical Image Analyze Group | Class-centroid-Based Openset Classfication method on Skin Lession | 0.557 |
| MelanoNorm_IITRopar<br>Indian Institute of Technology Ropar | Classification Using Stacking and Long-Tail Distribution: High Confidence Interval | 0.546 |
| MH_team | Softmax Ensemble Model and Sigmoid Ensemble Model | 0.544 |
| BGU_hackers<br>Ben Gurion University | AAAR Ensemble pruning Approach | 0.543 |
| SabanciUnivTeam<br>Sabanci University | Submission 3 - Anomaly Detection | 0.533 |
| deltamicro<br>Delta Micro Technology Inc | transferlearning-ensemble-averaging | 0.532 |
| offer_show<br>iSee-SYSU | Ensemble of model 1 and model 2 | 0.532 |
| MIP<br>Southern Medical University | ensemble three ResNext50 with SE Block | 0.531 |
| Tencent Medical AI Lab | Top models ensemble with threshold | 0.525 |
| bashiri | Deep Convolution Neural Network with data augmentation | 0.523 |
| Airdoctor | Embedding vectors and Ensemble models | 0.522 |
| One-Two-Three | VGG-16 for Skin Lesion Diagnosis | 0.519 |
| shallow learning<br>Institute of Automation, Chinese Academy of Sciences, Beijing 100190, China | Skin Lesion Analysis Towards Melanoma Detection using Deep Neural Networks | 0.518 |
| VisinVis<br>ETRI | Ensembled Transfer Neural Networks by using Lesion Correlation Learning : Approach 3 | 0.513 |
| cu<br>cuhk | Ensembles with external data | 0.510 |
| CureSkinAI<br>CureSkin | Convolutional Neural Networks Ensemble towards Skin Lesion Analysis of Dermoscopic Images | 0.509 |
| Predicthy LLC | Take max value among the combination of four neural networks | 0.507 |
| Hsinwei | Convolutional Ensemble with Out-of-Distribution Detector | 0.505 |
| MMU-VCLab<br>Manchester Metropolitan University | Two-stage Ensemble Method | 0.502 |
| Yongsheng Pan | FV-RES | 0.501 |
| Pan Galactic | Deep Learning Resnet50 | 0.493 |
| SY2<br>Beihang University | Skin Lesion Diagnosis using color constancy and loss weighting with external dataset 03 | 0.492 |
| logreg | EfficientNet b1 with augmentations | 0.492 |
| MGI<br>National Institutes of Biotechnology Malaysia | Densenet161 with discriminative learning rate | 0.489 |
| gxl_xgy_llz_victory<br>Institute of automation, Chinese academy of sciences | Ensembled Model for Skin Lesion Classification | 0.489 |
| Mt.Smart<br>MTlab,Meitu Inc | Multiple Convolution Neural Net Ensemble | 0.482 |
| BMIT<br>Biomedical and Multimedia Information Technology, University of Sydney | 152 Layer Resnet | 0.481 |
| Aiden Gatani | densenet201 | 0.470 |
| Le-Health<br>Lenovo Research | ensemble strategy 2 | 0.469 |
| SRMC<br>DataGenius | Approch 1 : Usable Predictive Model - Densenet121 | 0.465 |
| SY1<br>Beihang university | Skin Lesion Diagnosis using discrimination criterion | 0.464 |
| Panetta's Vision and Sensing Systems Lab<br>Tufts University | One-class SVM pre-filter + VGG16 CNN | 0.445 |
| IML group - DFKI<br>Interactive Machine Learning (IML) - German Research Center for Artificial Intelligence (DFKI) | One-class SVM pre-filter + VGG16 CNN | 0.445 |
| SIGMA<br>NUST | Ensemble of Fine-tuned DNNs for Skin Lesion Image Classication | 0.438 |
| skychain | resnext50_32x4d | 0.437 |
| SJ_T1 | Ten1 | 0.429 |
| KDIS<br>University of Cordoba & Maimonides Biomedical Research Institute of Cordoba | Multi-view convolutional architecture - Margin sampling version | 0.429 |
| SUMMER | Feature Fusion for Accurate Skin Lesion Analysis | 0.427 |
| AIRL<br>Central South University | Skin Lesion Classification with Out-of-Distribution Detection Using Deep Neural Network Ensemble	 | 0.426 |
| sysutest1 | mcd1 | 0.423 |
| I don't know what to eat | SEnet154 | 0.419 |
| UH ML Lab<br>University of Hawaii at Manoa | Inceptionv3 with CBAM | 0.419 |
| LLCW<br>Institute of Automation, Chinese Academy of Sciences | InceptionV3 with Transfer Learning | 0.405 |
| sstl<br>上海计算机软件技术开发中心 | Fine-grained skin image classification | 0.370 |
| Cihan Soylu | Transfer learning with DenseNet201 | 0.370 |
| CrazyLearningTeam | CrazyLearningTeam | 0.346 |
| SB<br>EPFL | Skin Lesion Analysis Towards Melanoma Detection using Siamese neural network | 0.329 |
| SIBET CAS<br>Suzhou Institute of Biomedical Engineering and Technology, Chinese Academy of Sciences | Semi-Supervised GAN | 0.327 |
| UTHealth-Onto.<br>University of Texas Health Science Center at Houston | Random Forest Ensemble_xy | 0.315 |
| SharpestMinds | Data Augmentation and Transfer Learning with ResNet50 with Cyclical Learning Rates | 0.304 |
| SkinLegion<br>Persistent Systems Ltd | Baseline Classifier | 0.273 |
| mvlab-skin<br>Indian Institute of Technology Roorkee | ISIC 2019 : Deep Ensembled Framework for Skin Lesion Analysis Towards Melanoma Detection (DEFSMD) | 0.258 |
| YouMe AI | Ensembles of Deep Convolution Neural Networks - updated | 0.251 |
| IT Derm Lab<br>DBE | Fine tune Resnet | 0.109 |
| jasdeep<br>Singh | Asymmetrical loss function and cutout along the corners | 0.108 |
| TUKL<br>NUST | ResNet50 ConvNet using cyclical learning rates and transfer learning | 0.106 |
| P | ResNet50 | 0.048 |


<br><br>

---
### Discussion


<br><br>

---
### Limitations


<br><br>

---
### Conclusion & Future Work






<!--
<br><br>

---
### References

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

-->






