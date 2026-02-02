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

## Environment Installation
`requirements.txt` contains the minimum required packages to train and inference Stylos.

## Dataset Preparation
You might want to change the dataset location in the following config files, `config/dataset/co3d.yaml`, `config/dataset/dl3dv.yaml`, and `config/dataset/dl3dv960.yaml`

## Training Instructions
We first train Stylos on the DL3DV dataset with Jittor color augmentations to learn geometry-related knowledge.
```bash
python src/main.py +experiment=dl3dv_geo trainer.num_nodes=1
```
After that, we load the pre-trained Stylos weights obtained from the previous step and further train the model for style learning,
```bash
python src/main.py +experiment=dl3dv_style trainer.num_nodes=1
```
Note: We have trained multiple versions of Stylos on DL3DV. As a result, certain training settings, e.g., loss configurations, may not exactly match those used to produce the released model weights.

## Quick Inference

For fast testing and inference only, please switch to:

👉 **[`quick_inference`](https://github.com/hanzhouliu/StylOS/tree/quick_inference)**

This branch contains the model weight that is for demo only.

## Timeline & TODO
The complete codebase will be **fully released soon**. We appreciate your patience and interest. Thanks for your attention and support! 

- [x] **Sep 2025** — Paper available on **arXiv**
- [x] **Jan 2026** — Paper accepted by **ICLR 2026**
- [x] **Jan 2026** — Hugging Face demo released
- [x] **Jan 2026** — Inference pipeline released  
  (please refer to the [`quick_inference` branch](https://github.com/hanzhouliu/StylOS/tree/quick_inference))
- [ ] — Release full training code (working on...(:3_ヽ)_)
- [ ] — Release evaluation instructions and codes
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