# SFR-Net

<p align="center">
  <a href="https://arxiv.org/abs/2605.25737"><img src="https://img.shields.io/badge/arXiv-2605.25737-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/shadowwalk/SFR-Net"><img src="https://img.shields.io/badge/Hugging%20Face-Weights-FFD21E.svg" alt="Hugging Face weights"></a>
</p>


<p align="center">
  English | <a href="README_zh-CN.md">简体中文</a>
</p>


<p align="center">
  <img src="pics/SFR-Net-cover.png" alt="SFR-Net cover" width="100%">
</p>


<h2 align="center">
  Learning Scale-Frustum Representations for Ultra-Wide Area<br>
  Remote Sensing Image Segmentation
</h2>


## Overview 🧭

SFR-Net is designed for semantic segmentation of ultra-wide area (UWA) remote sensing images, where both the pixel count and geographical coverage are extremely large. It constructs aligned local, short-range, and long-range observations around the same Projection Reference Point (PRP), resizes them to a unified input size, and distinguishes them with learnable scale embeddings. A Cascaded Cross-Scale Fusion (CCSF) module then injects contextual information into the local representation progressively, preserving fine details while improving long-range semantic continuity.

<p align="center">
  <img src="pics/sfrnet-framework.png" alt="Overall framework of SFR-Net" width="100%">
</p>


## News 📰

- **2026-08-26:** We updated the codebase, fixed known bugs, improved the inference, testing, and visualization scripts, and released trained weights for GID, FBPS, and Inria Aerial.
- **2026-07-11:** We received the first-round review decision from IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS), and the manuscript was invited for major revision.
- **2026-05-25:** Our paper, [“SFR-Net: Learning Scale-Frustum Representations for Ultra-Wide Area Remote Sensing Image Segmentation”](https://arxiv.org/abs/2605.25737), was released on arXiv.
- **2026-05-20:** Our paper, “SFR-Net: Learning Scale-Frustum Representations for Ultra-Wide Area Remote Sensing Image Segmentation,” was submitted to IEEE TGRS.
- **2026-05-11:** We released the initial code version with training and testing scripts and pretrained weights.

## Highlights ✨

- We formulate ultra-wide area remote sensing image segmentation as a task that jointly considers large pixel counts, extremely wide geographical coverage, significantly varying object scales, and long-range semantic continuity.
- Scale-Frustum Representations unify local, short-range, and long-range observations around the same PRP. The released GID/FBPS configs use distances `[1, 3, 14]`, while the Inria Aerial config uses `[1, 3, 10]`.
- Learnable scale embeddings explicitly identify resized observations from different spatial ranges.
- The CCSF module progressively introduces nearby and broader contextual cues into detailed local features.
- SFR-Net achieves state-of-the-art results on the UWA GID and FBPS benchmarks. The SFR representation can also improve the accuracy and convergence speed of generic segmentation networks.

## Performance 📊

The following table is taken from the paper. SFR-Net reaches `74.67%` mIoU on GID and `77.24%` mIoU on FBPS in the paper setting.

<p align="center">
  <img src="pics/sfrnet-performance.png" alt="Quantitative comparison on GID and FBPS" width="100%">
</p>


## Repository Layout 🗂️

```text
SFR-Net/
├── configs/
│   ├── _base_/
│   │   ├── datasets/
│   │   ├── schedules/
│   │   └── default_runtime.py
│   ├── gid/sfrnet_swinl_320k_gid.py
│   ├── fbps/sfrnet_swinl_320k_fbps.py
│   └── inria_aerial/sfrnet_swinl_320k_inria_aerial.py
├── mmseg/
│   ├── datasets/transforms/sfr_loading.py
│   ├── datasets/uwa_dataset.py
│   ├── models/backbones/sfr_net.py
│   └── models/necks/ccsf_neck.py
├── tools/
│   ├── train.py
│   ├── test.py
│   ├── sfr_inference.py
│   ├── get_res_iou.py
│   └── visualizer.py
├── pics/
├── pretrain/
├── weights/
├── README.md
└── README_zh-CN.md
```

The release keeps the default SFR-Net pathway and the GID, FBPS, and Inria Aerial configurations. Multi-distance ablations and other experimental-only modules are intentionally excluded.

## Weights 🔑

All pretrained backbones and released SFR-Net checkpoints are hosted in the [SFR-Net Hugging Face repository](https://huggingface.co/shadowwalk/SFR-Net).

### Available files

| Type                                | File                                                         | Expected location                                            |
| ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ResNet-18 ImageNet pretraining      | [`resnet18_v1c-b5776b93.pth`](https://huggingface.co/shadowwalk/SFR-Net/blob/main/pretrain/resnet18_v1c-b5776b93.pth) | `pretrain/resnet18_v1c-b5776b93.pth`                         |
| Swin-Large ImageNet-22K pretraining | [`swin_large_patch4_window12_384_22k_20220412-6580f57d.pth`](https://huggingface.co/shadowwalk/SFR-Net/blob/main/pretrain/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth) | `pretrain/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth` |
| GID checkpoint                      | [`iter_320000_gid.pth`](https://huggingface.co/shadowwalk/SFR-Net/blob/main/weights/iter_320000_gid.pth) | `weights/iter_320000_gid.pth`                                |
| FBPS checkpoint                     | [`iter_320000_fbps.pth`](https://huggingface.co/shadowwalk/SFR-Net/blob/main/weights/iter_320000_fbps.pth) | `weights/iter_320000_fbps.pth`                               |
| Inria Aerial checkpoint             | [`iter_320000_inria.pth`](https://huggingface.co/shadowwalk/SFR-Net/blob/main/weights/iter_320000_inria.pth) | `weights/iter_320000_inria.pth`                              |

You can download the files with the Hugging Face CLI:

```bash
pip install -U huggingface_hub
hf download shadowwalk/SFR-Net --local-dir downloads/SFR-Net
cp -r downloads/SFR-Net/pretrain/. pretrain/
cp -r downloads/SFR-Net/weights/. weights/
```

### Released checkpoint results

| Dataset      | OA (%) | mIoU (%) | mF1 (%) | Checkpoint                      |
| ------------ | -----: | -------: | ------: | ------------------------------- |
| GID          |  86.82 |    74.46 |   85.73 | `weights/iter_320000_gid.pth`   |
| FBPS         |  93.50 |    77.86 |   66.72 | `weights/iter_320000_fbps.pth`  |
| Inria Aerial |  96.91 |   83.96* |  91.28* | `weights/iter_320000_inria.pth` |

`*` For Inria Aerial, IoU and F1 are reported for the building class only. The released checkpoints were trained with random seed `42`; their results therefore differ slightly from the values reported in the paper.

The backbone paths are currently defined in `mmseg/models/backbones/sfr_net.py`. No code change is required if the two pretrained files are kept under `pretrain/` and commands are executed from the repository root.

## Installation 🛠️

Create an environment with a PyTorch/CUDA combination suitable for your GPU, then install SFR-Net from the repository root:

```bash
conda create -n sfrnet python=3.10 -y
conda activate sfrnet

# Install PyTorch first according to https://pytorch.org/get-started/locally/
pip install -U openmim
mim install mmengine "mmcv>=2.0.0"
pip install -r requirements.txt
pip install -v -e .
pip install mxnet
```

`mxnet` is used by `tools/sfr_inference.py` to read the original ultra-wide images.

## Data Preparation 🗃️

Official dataset pages:

| Dataset      | Website                                                      |
| ------------ | ------------------------------------------------------------ |
| GID          | [Gaofen Image Dataset](https://x-ytong.github.io/project/GID) |
| FBPS         | [Five-Billion-Pixels](https://x-ytong.github.io/project/Five-Billion-Pixels.html) |
| Inria Aerial | [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/) |

Organize the datasets as follows:

```text
SFR-Net/
└── data/
    ├── GID/
    │   ├── Image_train/
    │   ├── Image_test/
    │   ├── annos_train_5l/
    │   ├── annos_test_5l/
    │   ├── annos_train_24l/
    │   └── annos_test_24l/
    └── inria_aerial/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── Label/
            ├── train/
            ├── val/
            └── test/
```

GID and FBPS use the same GF-2 images but different label folders. GID uses the 5-category annotations and produces 6 class indices including background; FBPS uses the 24-category annotations and produces 25 class indices including background. Inria Aerial uses two class indices: background and building.

The released configs still contain the original local absolute paths. Before training or validation, update these three files:

```python
# configs/_base_/datasets/gid.py
data_root = 'data/GID'

# configs/_base_/datasets/fbps.py
data_root = 'data/GID'

# configs/_base_/datasets/inria_aerial.py
data_root = 'data/inria_aerial'
```

Alternatively, keep the datasets elsewhere and set each `data_root` to the corresponding absolute path. The folder names below `data_root` must still match the structure shown above.

## Training 🏋️

Before training:

1. Set `data_root` in the appropriate file under `configs/_base_/datasets/` as described in Data Preparation.
2. Check `batch_size` and `num_workers` in the selected experiment config. The released configs use batch size `4` and override `num_workers` to `64`; reduce them if your GPU memory or CPU resources are limited.
3. Keep the two backbone checkpoints under `pretrain/`, or update `depth2ckpt` in `mmseg/models/backbones/sfr_net.py` if you use different locations.

Train with random seed `42` (the default in `configs/_base_/default_runtime.py` and `tools/train.py`):

```bash
python tools/train.py configs/gid/sfrnet_swinl_320k_gid.py \
  --work-dir work_dirs/gid

python tools/train.py configs/fbps/sfrnet_swinl_320k_fbps.py \
  --work-dir work_dirs/fbps

python tools/train.py configs/inria_aerial/sfrnet_swinl_320k_inria_aerial.py \
  --work-dir work_dirs/inria_aerial
```

Add `--amp` to enable automatic mixed precision. Use `--resume` with the same `--work-dir` to continue from its latest checkpoint.

## Inference 🛰️

`tools/sfr_inference.py` contains original-machine defaults in the `DATASETS` dictionary, including `/mnt/dataset/zhongchuyu/...`. Either replace the `src` entries with `data/GID/Image_test` and `data/inria_aerial/images/test`, or pass `--src` explicitly as shown below. Command-line values take precedence over those defaults.

```bash
python tools/sfr_inference.py \
  --dataset gid \
  --src data/GID/Image_test \
  --dst work_dirs/gid_predictions \
  --config configs/gid/sfrnet_swinl_320k_gid.py \
  --ckpt weights/iter_320000_gid.pth \
  --stride 128

python tools/sfr_inference.py \
  --dataset fbps \
  --src data/GID/Image_test \
  --dst work_dirs/fbps_predictions \
  --config configs/fbps/sfrnet_swinl_320k_fbps.py \
  --ckpt weights/iter_320000_fbps.pth \
  --stride 128

python tools/sfr_inference.py \
  --dataset inria_aerial \
  --src data/inria_aerial/images/test \
  --dst work_dirs/inria_aerial_predictions \
  --config configs/inria_aerial/sfrnet_swinl_320k_inria_aerial.py \
  --ckpt weights/iter_320000_inria.pth \
  --stride 128
```

The default `--load-type random` builds the complete scale-frustum representation. Predictions are saved as single-channel class-index PNG masks.

## Metrics and Visualization 🎨

### Metrics

`tools/get_res_iou.py` currently stores the original ground-truth paths in its `DATASETS` dictionary and does not provide a `--gt` argument. Update that dictionary before evaluation:

```python
DATASETS = {
    'gid': ('data/GID/annos_test_5l', 6),
    'fbps': ('data/GID/annos_test_24l', 25),
    'inria_aerial': ('data/inria_aerial/Label/test', 2),
}
```

Then compute the metrics:

```bash
python tools/get_res_iou.py --dataset gid \
  --pred work_dirs/gid_predictions

python tools/get_res_iou.py --dataset fbps \
  --pred work_dirs/fbps_predictions

python tools/get_res_iou.py --dataset inria_aerial \
  --pred work_dirs/inria_aerial_predictions
```

### Visualization

`tools/visualizer.py` has no fixed dataset path; provide the input and output directories on the command line. Its `PALETTES` dictionary contains the GID, FBPS, and Inria Aerial color maps and only needs modification if your class-index convention changes.

```bash
python tools/visualizer.py --dataset gid \
  --src work_dirs/gid_predictions \
  --dst work_dirs/gid_visualizations

python tools/visualizer.py --dataset fbps \
  --src work_dirs/fbps_predictions \
  --dst work_dirs/fbps_visualizations

python tools/visualizer.py --dataset inria_aerial \
  --src work_dirs/inria_aerial_predictions \
  --dst work_dirs/inria_aerial_visualizations
```

## Contact ✉️

If you find this work useful, please cite our [paper](https://arxiv.org/abs/2605.25737):

```bibtex
@article{zhong2026sfr,
  title={SFR-Net: Learning Scale-Frustum Representations for Ultra-Wide Area Remote Sensing Image Segmentation},
  author={Zhong, Chuyu and Chen, Keyan and Yang, Qinzhe and Chen, Bowen and Zou, Zhengxia and Shi, Zhenwei},
  journal={arXiv preprint arXiv:2605.25737},
  year={2026}
}
```

Questions and bug reports are welcome at **buaazcy@buaa.edu.cn**.

If you find this repository helpful, please give it a star. Finally, here is Phoebe. You are not allowed to bully her.

<p align="left">
  <img src="pics/phoebe.png" width="300" alt="Phoebe">
</p>

