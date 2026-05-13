# SFR-Net 🌍✨

**Learning Scale-Frustum Representations for Ultra-Wide Area Remote Sensing Image Segmentation**

This repository is a lightweight, GitHub-ready implementation of **SFR-Net**
for ultra-wide area (UWA) remote sensing image segmentation. It follows the
MMSegmentation-style project layout, but keeps only the default SFR-Net setting
used for GID and FBPS experiments. Clean, focused, and ready to run. 🚀

## What Is Inside? 🧭

SFR-Net targets remote sensing images with both:

- huge pixel counts, and
- extremely wide geographical coverage.

The core idea is **Scale-Frustum Representations (SFR)**: for each Projection
Reference Point (PRP), the model observes the image from multiple distances,
such as `[1, 3, 14]`, then resizes those observation windows to the same
`512 x 512` resolution. This lets the network see local details, short-range
context, and long-range semantic continuity at the same time.

## Highlights ⭐

- **SFR-Net** backbone for ultra-wide area segmentation.
- **Scale-Frustum Representations** with default distances `[1, 3, 14]`.
- **ScaleEmbeddings** to distinguish observation windows from different
  distances.
- **Swin-Large** as the main encoder for local observation.
- **ResNet-18** sub-encoders for short-range and long-range observation.
- **Cascaded Cross-Scale Fusion (CCSF)** for injecting contextual features.
- **UPerNet** main decoder and **FCNHead** auxiliary decoder.
- Default configs for **GID** and **FBPS** only.
- Pretrained weights are read from the local `pretrain/` folder, so training is
  reproducible even when the network is feeling dramatic. 🎒

## Repository Layout 📦

```text
SFR-Net/
├── configs/
│   ├── gid/sfrnet_swinl_320k_gid.py
│   ├── fbps/sfrnet_swinl_320k_fbps.py
│   └── _base_/
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
└── README.md
```

The 4-distance, 5-distance, and ablation configs are intentionally not included
in this release. This repo focuses on the default `[4, 18, 18]` setup:
Swin-Large main encoder plus two ResNet-18 sub-encoders.

## Pretrained Weights 🔑

SFR-Net expects ImageNet pretrained weights at the following local paths:

```text
pretrain/
├── resnet18_v1c-b5776b93.pth
└── swin_large_patch4_window12_384_22k_20220412-6580f57d.pth
```

If you need the Swin-Large and ResNet-18 pretrained weights, they can be
obtained from this Google Drive folder:
[SFR-Net pretrained weights](https://drive.google.com/drive/folders/1Q71Hwb9KvcarJYGL9uB-AsidpsGBDLyr?usp=drive_link).
Drop the two `.pth` files into `pretrain/`, and the default configs will pick
them up directly.

## Installation 🛠️

Create a Python environment with PyTorch and CUDA that matches your machine,
then install the dependencies:

```bash
pip install -U openmim
mim install mmengine "mmcv>=2.0.0"
pip install -r requirements.txt
pip install -v -e .
```

The SFR input loader uses `mxnet` for image reading. If your environment does
not already have it:

```bash
pip install mxnet
```

## Data Preparation 🗂️

By default, configs expect:

```text
data/GID/
├── Image_train/
├── Image_test/
├── annos_train_5l/
├── annos_test_5l/
├── annos_train_24l/
└── annos_test_24l/
```

You can either place data there directly or make a symbolic link:

```bash
mkdir -p data
ln -s /path/to/GID data/GID
```

- **GID** uses `annos_*_5l` and 6 output classes.
- **FBPS** uses `annos_*_24l` and 25 output classes.

## Training 🏃

Train SFR-Net on GID:

```bash
python tools/train.py configs/gid/sfrnet_swinl_320k_gid.py
```

Train SFR-Net on FBPS:

```bash
python tools/train.py configs/fbps/sfrnet_swinl_320k_fbps.py
```

Default training uses AdamW, a 1500-iteration warm-up, PolyLR, batch size 4,
and 320k maximum iterations.

## Testing 🧪

```bash
python tools/test.py configs/gid/sfrnet_swinl_320k_gid.py work_dirs/gid/iter_320000.pth
```

```bash
python tools/test.py configs/fbps/sfrnet_swinl_320k_fbps.py work_dirs/fbps/iter_320000.pth
```

## Ultra-Wide Area Inference 🛰️

Run scale-frustum sliding-window inference on GID:

```bash
python tools/sfr_inference.py \
  --dataset gid \
  --src data/GID/Image_test \
  --dst work_dirs/gid_predictions \
  --config configs/gid/sfrnet_swinl_320k_gid.py \
  --ckpt work_dirs/gid/iter_320000.pth \
  --stride 128
```

Run it on FBPS:

```bash
python tools/sfr_inference.py \
  --dataset fbps \
  --src data/GID/Image_test \
  --dst work_dirs/fbps_predictions \
  --config configs/fbps/sfrnet_swinl_320k_fbps.py \
  --ckpt work_dirs/fbps/iter_320000.pth \
  --stride 128
```

Each PRP builds Scale-Frustum Representations with distances `[1, 3, 14]`,
predicts the local observation, and stitches predictions over the whole UWA
image. Tiny windows, big picture. 🌌

## Metrics and Visualization 🎨

Compute IoU:

```bash
python tools/get_res_iou.py --dataset gid --pred work_dirs/gid_predictions
python tools/get_res_iou.py --dataset fbps --pred work_dirs/fbps_predictions
```

Colorize masks:

```bash
python tools/visualizer.py --dataset gid --src work_dirs/gid_predictions --dst work_dirs/gid_vis
python tools/visualizer.py --dataset fbps --src work_dirs/fbps_predictions --dst work_dirs/fbps_vis
```

## Contact 💌

Questions, bugs, training adventures, or SFR-Net stories are welcome:

**buaazcy@buaa.edu.cn**

Finally, here's Phoebe. You're not allowed to bully her.

<p align="left">
<img src="./pics/phoebe.jpg" width="300">
</p>
