<div align="center">

# 🎨 Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting

[![arXiv](https://img.shields.io/badge/arXiv-2509.26455-b31b1b.svg)](https://arxiv.org/abs/2509.26455)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Stylos%20Checkpoint-ffd21f.svg?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/HanzhouLiu/Stylos_Weights)

*<a href="https://hanzhouliu.github.io/">Hanzhou Liu</a>\*, 
<a href="https://scholar.google.com/citations?user=5F41hjgAAAAJ&hl=en">Jia Huang</a>, 
<a href="https://engineering.tamu.edu/electrical/profiles/mlu.html">Mi Lu</a>, 
<a href="https://engineering.tamu.edu/mechanical/profiles/saripalli.html">Srikanth Saripalli</a>, 
<a href="https://scholar.google.com/citations?user=jW34BjIAAAAJ&hl=en">Peng Jiang</a>\*†*  

<sub>\* Equal Contribution † Corresponding Author</sub>

</div>

Stylos turns casually captured multi-view photos or monocular video frames into stylized 3D Gaussian splats with a single forward pass. This repository contains the inference pipeline used in our paper along with a lightweight Gradio demo for interactive exploration—ready for you to play, remix, and share! 🥳

---

## Highlights ✨
- Single-pass 3D stylization for image collections or videos with automatic frame extraction.
- Pretrained Gaussian Splatting decoder packaged for `Stylos.from_pretrained`.
- CLI saves stylized RGB/depth videos and a shareable Gaussian `.ply`.
- Optional Gradio UI for quick experimentation without writing code.

## Repository Status 🚀
- The public release currently focuses on inference, visualization, and demo utilities.
- Additional tooling and training code will follow; watch the repo to stay updated.
- Questions or feedback? Open an issue or email Hanzhou Liu at `hanzhou1996@tamu.edu`. We love hearing from fellow creators! 💌

## Table of Contents
- [Getting Started](#getting-started)
- [Model Checkpoints](#model-checkpoints)
- [CLI Inference](#cli-inference)
- [Gradio Demo](#gradio-demo)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Getting Started

### Prerequisites
- Python 3.10 or newer.
- At least, PyTorch 2.2.0 (CUDA 12.1) for the reference environment; other recent PyTorch/CUDA combos also work.
- An NVIDIA GPU is recommended for real-time performance, but CPU fallbacks are available.

### Clone the Repository
```bash
git clone https://github.com/HanzhouLiu/StylOS.git
cd StylOS
git checkout quick_inference
```

### Create an Environment
Below is an example using Conda; feel free to adapt it to virtualenv or another workflow.
```bash
conda create -y -n stylos python=3.10
conda activate stylos
```

### Install Dependencies
[Optional] If you would like to install PyTorch manually, please ensure the correct CUDA wheels are used and run the following example command
```
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```
Otherwise, install all packages in one line of commands.
```bash
pip install -r requirements.txt
```
`requirements.txt` contains the minimal set for inference and the demo. 

For cuda-compiled version of RoPE2D, run following command
```bash
cd src/model/encoder/backbone/croco/curope
pip install .
```

---

## Model Checkpoints 🧳

Stylos checkpoints follow the Hugging Face `PyTorchModelHubMixin` format. This repo ships with a ready-to-use checkpoint in `stylos_hf_chk/`, so the quickest path is simply:
```bash
python stylos_infer.py --model stylos_hf_chk ...
```

To fetch the latest weights manually (or if you removed the local copy), download them from [Hugging Face](https://huggingface.co/datasets/HanzhouLiu/Stylos_Weights):
```bash
huggingface-cli download HanzhouLiu/Stylos_Weights --repo-type dataset --local-dir stylos_hf_chk --exclude ".gitattributes"
```
The directory, i.e., stylos_hf_chk/DL3DV/2025-10-09_16-10-03, must contain `config.json` and `model.safetensors` so that `Stylos.from_pretrained()` can resolve the model.

---

## CLI Inference

`stylos_infer.py` runs Stylos on either a folder of pre-extracted images or a raw video file. Outputs are written under `output/inference/<style>__<sequence>/`.

### Multi-View Photo Collections
```bash
python stylos_infer.py \
  --model stylos_hf_chk/DL3DV/2025-10-09_16-10-03 \
  --image-folder examples/TNT/Garden \
  --style-image examples/demo_styles/977.png \
  --num-views 4 \
  --output-dir output/inference \
  --precision auto \
  --device auto
```

### Video Input (Automatic 1 FPS Sampling)
```bash
python stylos_infer.py \
  --model stylos_hf_chk \
  --video examples/video/fox.mp4 \
  --style-image examples/demo_styles/00047052.png \
  --output-dir output/inference \
  --extract-fps 1 \
  --max-seconds 60 \
  --keep-frames
```
When `--video` is used, frames are temporarily extracted to a scratch directory. Pass `--keep-frames` to inspect them afterward—and maybe make a behind-the-scenes GIF! 🎞️

### Common Flags
| Flag | Description |
| --- | --- |
| `--num-views` | Cap the number of input frames (default `-1`, meaning all available frames). |
| `--device` | Choose `auto`, `cuda`, or `cpu`. `auto` prefers CUDA if present. |
| `--precision` | Select `auto`, `fp32`, `fp16`, or `bf16`. |
| `--seed` | Optional RNG seed for reproducibility. |
| `--output-dir` | Base directory for all exports (defaults to `output/inference`). |

### Output Artifacts
- `rgb.mp4`: Stylized novel-view video rendered from interpolated camera poses.
- `depth.mp4`: Colored depth visualization aligned with the RGB sequence.
- `gaussians.ply`: 3D Gaussian Splat point cloud for downstream viewers.
- Intermediate frame dumps (optional) when `--keep-frames` is active.

---

## Gradio Demo

Launch the interactive UI to try Stylos in a browser:
```bash
python demo_gradio.py
```

The app downloads the checkpoint the first time it runs (if not already cached), then exposes:
- Drag-and-drop upload for videos or image sets.
- Style image selection and preset examples.
- Real-time rendering preview plus download links to RGB/depth videos and the `.ply`.
- A friendly playground to iterate on stylistic ideas in minutes! 🌈

---

## Citation

If you use Stylos in your research, please cite:

```bibtex
@misc{liu2025stylosmultiview3dstylization,
      title={Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting},
      author={Hanzhou Liu and Jia Huang and Mi Lu and Srikanth Saripalli and Peng Jiang},
      year={2025},
      eprint={2509.26455},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.26455},
}
```

---

## Acknowledgements 🙏

Stylos builds upon excellent open-source libraries, including [AnySplat](https://github.com/InternRobotics/AnySplat), [VGGT](https://github.com/facebookresearch/vggt), [NoPoSplat](https://github.com/cvg/NoPoSplat), [CUT3R](https://github.com/CUT3R/CUT3R/tree/main), and [gsplat](https://github.com/nerfstudio-project/gsplat). We sincerely thank the authors of these projects for releasing their code.

## License 📜

Released under the [MIT License](LICENSE).