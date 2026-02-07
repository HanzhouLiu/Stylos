<div align="center">

## Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting

[![arXiv](https://img.shields.io/badge/arXiv-2509.26455-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2509.26455) <a href="https://hanzhouliu.github.io/Stylos"><img src="https://img.shields.io/badge/Project_Page-green?style=flat-square" alt="Project Page"></a> <a href="https://huggingface.co/spaces/HanzhouLiu/Stylos_Gradio"> <img src="https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow?style=flat-square"> </a>


*<a href="https://hanzhouliu.github.io/">Hanzhou Liu</a>\*, 
<a href="https://scholar.google.com/citations?user=5F41hjgAAAAJ&hl=en">Jia Huang</a>, 
<a href="https://engineering.tamu.edu/electrical/profiles/mlu.html">Mi Lu</a>, 
<a href="https://engineering.tamu.edu/mechanical/profiles/saripalli.html">Srikanth Saripalli</a>, 
<a href="https://scholar.google.com/citations?user=jW34BjIAAAAJ&hl=en">Peng Jiang</a>\*†*  

<sub>\* Equal Contribution † Corresponding Author</sub>

</div>

## Table of Contents

- [Overview](#overview)
- [Full Instructions](#full-instructions)
  - [Environment Setup](#1-environment-setup)
  - [Dataset Preparation](#2-dataset-preparation)
    - [CO3D Dataset](#21-co3d-dataset)
    - [DL3DV Dataset](#22-dl3dv-dataset)
  - [Training Guidelines](#3-training-guidelines)
    - [Train Stylos on CO3D](#31-train-stylos-on-co3d)
    - [Train Stylos on DL3DV](#32-train-stylos-on-dl3dv)
  - [Inference and Evaluation](#4-inference-and-evaluation)
    - [Test Data and Model Checkpoints](#41-test-data-and-model-checkpoints)
    - [Inference](#42-inference)
    - [Evaluation](#43-evaluation)
    - [Comparison with SOTA Methods](#44-comparison-with-sota-methods)
- [Quick Inference](#quick-inference)

## Overview
**TL;DR.** Stylos a single-forward 3D Gaussian framework for 3D style transfer that operates on unposed content, from a single image to a multi-view collection, conditioned on a separate reference style image.

## Full Instructions
### 1. Environment Setup
`requirements.txt` contains the minimum required packages to train and inference Stylos, which can be installed by running
```bash
pip install -r requirements.txt
```
For cuda-compiled version of RoPE2D, run following command
```bash
cd src/model/encoder/backbone/croco/curope
pip install .
```
### 2. Dataset Preparation
To modify the dataset location, please edit the following config files, [`config/dataset/co3d.yaml`](https://github.com/HanzhouLiu/Stylos/blob/main/config/dataset/co3d.yaml), [`config/dataset/dl3dv.yaml`](https://github.com/HanzhouLiu/Stylos/blob/main/config/dataset/dl3dv.yaml), [`config/dataset/dl3dv960.yaml`](https://github.com/HanzhouLiu/Stylos/blob/main/config/dataset/dl3dv960.yaml).

#### 2.1. CO3D Dataset
To download the CO3D dataset, run the following command,
```bash
bash scripts/sh_files/datasets/download_co3d.sh
```
To preprocess the CO3D dataset, run the following command without `--dry_run`,
```bash
python3 -m scripts.python_files.co3d_dataset_preprocess --co3d_root datasets/CO3D --dry_run
```
Note: Since Stylos is based on [Anysplat](https://github.com/InternRobotics/AnySplat) and [VGGT](https://github.com/facebookresearch/vggt), it does not require image poses during both training and inference.

#### 2.2. DL3DV Dataset
Please download the [DL3DV](https://github.com/DL3DV-10K/Dataset) dataset from their official website.

### 3. Training Guidelines
Please download the checkpoints from [huggingface link to Stylos](https://huggingface.co/datasets/HanzhouLiu/Stylos) to `checkpoints` in the current workingspace. We use the pre-trained VGG weights for computing losses.

#### 3.1. Train Stylos on CO3D
We first train Stylos to learn geometry-related knowledge, on 8 NVIDIA H200 GPUs.
```bash
python src/main.py +experiment=co3d_geo_global_base trainer.num_nodes=1
```
After that, we load the pre-trained Stylos weights obtained from the previous step and further train the model for style learning, on 4 NVIDIA GH200 GPUs.
```bash
python src/main.py +experiment=co3d_style_3d_loss_4gpus trainer.num_nodes=1
```
#### 3.2. Train Stylos on DL3DV
We first train Stylos to learn geometry-related knowledge.
```bash
python src/main.py +experiment=dl3dv_geo trainer.num_nodes=1
```
After that, we load the pre-trained Stylos weights obtained from the previous step and further train the model for style learning,
```bash
python src/main.py +experiment=dl3dv_style trainer.num_nodes=1
```
Note: We have trained multiple versions of Stylos on DL3DV, and released two of them. Certain training settings, e.g., loss configurations, number of iterations, number of view, and etc., may vary. Please adust the training hyperparamters according to your needs. In the released training codes, only Wikiart is supported as the style reference while DELAUNAY could be added easily.

### 4. Inference and Evaluation
#### 4.1. Test Data and Model Checkpoints 
The 50 test styles are located in [`examples/styles`](https://github.com/HanzhouLiu/Stylos/tree/main/examples/styles). Please copy that folder to the `datasets` directory.

Download all needed test data and checkpoints in [huggingface link to Stylos](https://huggingface.co/datasets/HanzhouLiu/Stylos).

Please use the [`output/exp_dl3dv_old`](https://huggingface.co/datasets/HanzhouLiu/Stylos/tree/main/output/exp_dl3dv_old). checkpoint to reproduce the quantative results of Stylos in the paper.


#### 4.2. Inference
To test Stylos on the CO3D dataset using a frame stride of 3, run:
```bash
bash scripts/sh_files/co3d_3d_loss/inference_frame_stride_3.sh
```
To test Stylos on the TNT dataset, run the following command,
```bash
scripts/sh_files/dl3dv2tnt/inference.sh 
```

#### 4.3. Evaluation
After inference, compute consistency metrics and Artscore on CO3D scenes by running,
```bash
scripts/sh_files/co3d_3d_loss/eval_frame_stride_3.sh 
```
To compute consistency metrics and Artscore of Stylos on each TNT scene, run,
```bash
scripts/sh_files/dl3dv2tnt/eval.sh 
```
The implementation of consistency metrics is modified from [StyleGaussian](https://github.com/Kunhao-Liu/StyleGaussian/issues/5#issuecomment-2078576765).

#### 4.4. Comparison with SOTA Methods
We reproduce or directly evaluate several 3D stylization models, which include [StyleGaussian](https://github.com/Kunhao-Liu/StyleGaussian/issues/5#issuecomment-2078576765), [G-Style](https://github.com/AronKovacs/g-style), [SGSST](https://github.com/JianlingWANG2021/SGSST), [StyleGaussian](https://github.com/Kunhao-Liu/StyleGaussian), and [Styl3R](https://github.com/WU-CVGL/Styl3R). We sincerely appreciate their open-source contributions to the 3D stylization community. The visual results are available at the following link, [huggingface link to SOTA comparisons](https://huggingface.co/datasets/HanzhouLiu/Stylos/tree/main/sota_comparisons).

## Quick Inference

For demo only, please switch to **[`quick_inference`](https://github.com/hanzhouliu/StylOS/tree/quick_inference)** branch.
To reproduce results in our paper, please use the **[`main`](https://github.com/HanzhouLiu/Stylos/tree/main)** branch.

## Timeline & TODO
The complete codebase will be **fully released soon**. We appreciate your patience and interest. Thanks for your attention and support! 

- [x] **Sep 2025** — Paper available on **arXiv**
- [x] **Jan 2026** — Paper accepted by **ICLR 2026**
- [x] **Jan 2026** — Hugging Face demo released
- [x] **Jan 2026** — Inference pipeline released  
  (please refer to the [`quick_inference` branch](https://github.com/hanzhouliu/StylOS/tree/quick_inference))
- [x] **Feb 2026** — Release full training code (still working on double-checking...(:3_ヽ)_)
- [x] **Feb 2026** — Release evaluation instructions and codes (still working on double-checking...(:3_ヽ)_)
- [x] **Feb 2026** — Release comparison results with state-of-the-art models
- [ ] — Paper final version available
---

⭐ If you find this project useful, please **give us a star** to help more people discover it.  
👀 You can also **watch the repository** (top-right corner) to stay updated on new features, papers, and releases.  

Your feedback and contributions are always welcome!
If you have any question, feel free to leave an issue or email Hanzhou Liu @ hanzhou1996@tamu.edu

## Citation
```bibtex
@article{liu2025stylos,
  title={Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting},
  author={Liu, Hanzhou and Huang, Jia and Lu, Mi and Saripalli, Srikanth and Jiang, Peng},
  journal={arXiv preprint arXiv:2509.26455},
  year={2025}
}
```