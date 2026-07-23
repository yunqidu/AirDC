<h2>
<a href="https://github.com/yunqidu/AirDC/" target="_blank">AirDC: Adaptive Iterative Depth Refinement Framework for Full-Range Metric Depth Completion</a>
</h2>

This is the official PyTorch implementation of **AirDC: Adaptive Iterative Depth Refinement Framework for Full-Range Metric Depth Completion**, published in *IEEE Transactions on Image Processing* (TIP), 2026.


## 🔍 Introduction

**AirDC** is an adaptive iterative depth refinement framework for full-range metric depth completion. The paper has been published in *IEEE Transactions on Image Processing* (TIP), Volume 35, pages 6658-6673, 2026.

- **Paper:** [IEEE Xplore](https://doi.org/10.1109/TIP.2026.3700921)
- **DOI:** [10.1109/TIP.2026.3700921](https://doi.org/10.1109/TIP.2026.3700921)

## 🔖 Citation

If this project is useful for research, please cite:

```bibtex
@article{shi2026airdc,
  title={AirDC: Adaptive Iterative Depth Refinement Framework for Full-Range Metric Depth Completion},
  author={Shi, Hongyu and Du, Yunqi and Zhang, Hongjuan and Li, Wenzhuo and Dong, Zhen and Li, Bijun and Tang, Luliang},
  journal={IEEE Transactions on Image Processing},
  volume={35},
  pages={6658--6673},
  year={2026},
  publisher={IEEE},
  doi={10.1109/TIP.2026.3700921}
}
```

## 💾 Datasets
We used three datasets (KITTI DC, Virtual KITTI 2.0, and MS2) for both training and evaluation.

The released configs use placeholder relative dataset roots:

```
./data/KITTI_Depth_Completion
./data/MS2
./data/virtual_kitti_v2
```

Before training or evaluation, either edit the YAML files directly or set dataset roots with environment variables:

```bash
export AIRDC_WORKSPACE=/path/to/AirDC
export KITTI_ROOT=/path/to/KITTI_Depth_Completion
export MS2_ROOT=/path/to/MS2
export VK2_ROOT=/path/to/virtual_kitti_v2
```

Dataset sources:

- KITTI DC: [KITTI Depth Completion](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)
- KITTI Raw: [KITTI Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php)
- Virtual KITTI 2.0: [Virtual KITTI 2.0](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2)
- MS2: [MS2 Dataset](https://sites.google.com/view/multi-spectral-stereo-dataset/home)

## 💻 Code

### Checkpoints

The following three checkpoints will be released before 2026-07-31. After release, download them from Google Drive and put them under the `log/` directory:

```text
AirDC/
├── log/
│   ├── kt_577.pth
│   ├── ms2_972.pth
│   └── vk2_2424.pth
```

Save converted public AirDC checkpoints under `log/ckpt/`:

```text
AirDC/
├── log/
│   ├── ckpt/
│   │   ├── kt_577_airdc.pth
│   │   ├── ms2_972_airdc.pth
│   │   └── vk2_2424_airdc.pth
```

The default configs load the following checkpoints:

| Dataset | Configs | Checkpoint |
| --- | --- | --- |
| KITTI DC | `train_all_att.yaml`, `val_all_att.yaml`, `test_all_att.yaml` | `log/kt_577.pth` |
| MS2 | `train_all_att_ms2.yaml`, `val_all_att_ms2.yaml` | `log/ms2_972.pth` |
| Virtual KITTI 2 | `train_all_att_vk2.yaml`, `val_all_att_vk2.yaml` | `log/vk2_2424.pth` |

For checkpoints stored elsewhere, either edit `ckpt_path` in the YAML config or override it from the command line.

To convert an old checkpoint to the public AirDC module layout, run:

```bash
python tools/convert_airdc_checkpoint.py \
  --workspace . \
  --config config/val_all_att.yaml \
  --input-ckpt log/kt_577.pth \
  --output-ckpt log/ckpt/kt_577_airdc.pth \
  --strip-module-prefix
```

### Environment Variables

Using environment variables makes the train/validation/test commands easier to reuse across devices:

```bash
export AIRDC_WORKSPACE=/path/to/AirDC
export KITTI_ROOT=/path/to/KITTI_Depth_Completion
export MS2_ROOT=/path/to/MS2
export VK2_ROOT=/path/to/virtual_kitti_v2
```

### Train

Train AirDC on KITTI DC:

```bash
bash train_bf16.sh
```

Train AirDC on MS2:

```bash
bash train_ms2.sh
```

Train AirDC on Virtual KITTI 2:

```bash
bash train_vk2.sh
```

The training scripts use the current repository directory as the workspace by default. Override it when code and logs are stored elsewhere:

```bash
AIRDC_WORKSPACE=/path/to/AirDC bash train_bf16.sh
```

Training outputs are saved to `log/<version>/`, where `version` is defined in the corresponding YAML config. For example, KITTI training uses `config/train_all_att.yaml` with `version: 0527_train`, so the outputs are saved under:

```text
log/0527_train/
├── latest_model.pth
├── best_model.pth
├── train_log.csv
├── val_log.csv
└── kittidc_AirDC_train_vis/
```

- `latest_model.pth`: checkpoint saved during training.
- `best_model.pth`: checkpoint with the best validation loss.
- `train_log.csv`: training metrics and hyperparameters.
- `val_log.csv`: validation metrics after each epoch.
- `*_train_vis/`: optional training visualizations when visualization is enabled.

### Validation

Validate AirDC on KITTI DC:

```bash
torchrun --nnodes=1 --nproc_per_node=1 run.py \
  --workspace $AIRDC_WORKSPACE \
  --log_dir log/ \
  --config_path val_all_att.yaml \
  --override \
    data_folder=$KITTI_ROOT \
    calib_folder=$KITTI_ROOT/calib \
    ckpt_path=ckpt/kt_577_airdc.pth \
  --no_debug
```

Validate AirDC on MS2:

```bash
torchrun --nnodes=1 --nproc_per_node=1 run.py \
  --workspace $AIRDC_WORKSPACE \
  --log_dir log/ \
  --config_path val_all_att_ms2.yaml \
  --override \
    data_folder=$MS2_ROOT \
    ckpt_path=ckpt/ms2_972_airdc.pth \
  --no_debug
```

Validate AirDC on Virtual KITTI 2:

```bash
torchrun --nnodes=1 --nproc_per_node=1 run.py \
  --workspace $AIRDC_WORKSPACE \
  --log_dir log/ \
  --config_path val_all_att_vk2.yaml \
  --override \
    data_folder=$VK2_ROOT \
    calib_folder=$KITTI_ROOT/calib \
    ckpt_path=ckpt/vk2_2424_airdc.pth \
  --no_debug
```

### Test

Run KITTI depth completion test with `test_all_att.yaml`:

```bash
torchrun --nnodes=1 --nproc_per_node=1 run.py \
  --workspace $AIRDC_WORKSPACE \
  --log_dir log/ \
  --config_path test_all_att.yaml \
  --override \
    data_folder=$KITTI_ROOT \
    calib_folder=$KITTI_ROOT/calib \
  --no_debug
```

The test config uses `split: test_completion` and loads `log/kt_577.pth` by default.

### Configs

The main configs are under `config/`. Edit these files to change dataset paths, checkpoints, and model/training hyperparameters:

| Dataset | Train config | Val config | Test config |
| --- | --- | --- | --- |
| KITTI DC | `config/train_all_att.yaml` | `config/val_all_att.yaml` | `config/test_all_att.yaml` |
| MS2 | `config/train_all_att_ms2.yaml` | `config/val_all_att_ms2.yaml` | - |
| Virtual KITTI 2 | `config/train_all_att_vk2.yaml` | `config/val_all_att_vk2.yaml` | - |

Important config fields:

| Field | Description |
| --- | --- |
| `split` | Running mode: `train`, `val`, or `test_completion`. |
| `version` | Output folder name under `log/`. |
| `dataset` | Dataset name: `kittidc`, `ms2`, or `vkitti2`. |
| `data_folder` | Dataset root path. |
| `calib_folder` | KITTI calibration folder. Required for KITTI DC and VKITTI2. |
| `ckpt_path` | Checkpoint filename or path. Relative paths are resolved under `log/`. |
| `epochs` | Number of training epochs. |
| `lr` | Learning rate. |
| `mixed_precision` | Whether to use mixed precision. |
| `num_iters` | Number of iterative refinement steps in AirDC. |
| `train_strategy` | Training strategy, e.g. `iSLDV`. |
| `lidar_lines` | Number of sparse LiDAR lines used as input. |
| `crop_height`, `crop_width` | Training crop size. |
| `oheight`, `owidth` | Validation/test crop size. |
| `vis_result` | Whether to save visualization results during validation. |

Config values can also be overridden directly from the command line:

```bash
torchrun --nnodes=1 --nproc_per_node=1 run.py \
  --workspace $AIRDC_WORKSPACE \
  --log_dir log/ \
  --config_path val_all_att.yaml \
  --override \
    data_folder=$KITTI_ROOT \
    calib_folder=$KITTI_ROOT/calib \
    num_iters=6 \
    lidar_lines=32 \
  --no_debug
```


## Dataset Layouts
We used three datasets (KITTI DC, Virtual KITTI 2.0, and MS2) for both training and evaluation.
### KITTI Depth Completion (KITTI DC)

KITTI DC dataset is available at the [KITTI DC Website](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion).
Color images are provided by the KITTI Raw dataset, available at the [KITTI Raw Website](http://www.cvlibs.net/datasets/kitti/raw_data.php).

The overall data directory is structured as follows:

```
├── kitti_depth
|   ├── calib
|   |  ├── 2011_09_26_calib_cam_to_cam.txt
|   |  ├── 2011_09_28_calib_cam_to_cam.txt
|   |  ├── 2011_09_30_calib_cam_to_cam.txt
|   |  ├── 2011_10_03_calib_cam_to_cam.txt
|   ├──data_depth_annotated
|   |  ├── train
|   |  ├── val
|   ├── data_depth_velodyne
|   |  ├── train
|   |  ├── val
|   ├── data_depth_selection
|   |  ├── test_depth_completion_anonymous
|   |  |── test_depth_prediction_anonymous
|   |  ├── val_selection_cropped
|   |  |  |── groundtruth_depth
|   |  |  |── image
|   |  |  |── intrinsics
|   |  |  |── velodyne_raw
├── kitti_raw
|   ├── 2011_09_26
|   ├── 2011_09_28
|   ├── 2011_09_29
|   ├── 2011_09_30
|   ├── 2011_10_03
```

### Virtual KITTI 2.0
Virtual KITTI 2.0 dataset is available at the [Virtual KITTI 2.0 Website](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2).
Virtual KITTI 2.0 dataset contains multiple scenes for training and testing.
- **Training**: `Scene01`, `Scene02`
- **Testing / Validation**: `Scene06`, `Scene18`, `Scene20`
```
├── vkitti2
|   ├── Scene01
|   |  ├── 15-deg-left
|   |  |  ├──── frames
|   |  |  |  ├── depth
|   |  |  |  |  ├── Camera_0
|   |  |  |  |  ├── Camera_1
|   |  |  |  ├── rgb
|   |  |  |  |  ├── Camera_0
|   |  |  |  |  ├── Camera_1
|   |  ├── ...
|   ├── Scene02
|   ├── Scene06
|   ├── Scene18
|   ├── Scene20
```

### MS2 Dataset
MS2 dataset is available at the [MS2 Website](https://sites.google.com/view/multi-spectral-stereo-dataset/home).
MS2 dataset contains multiple sequences for training and testing.
- **Training**: `_2021-08-06-11-23-45` (urban), `_2021-08-13-16-14-48` (Residential), `_2021-08-13-16-31-10` (road1), `_2021-08-13-17-06-04` (campus)
- **Testing / Validation**: `_2021-08-13-16-08-46` (road3)

The overall data directory is structured as follows:

```
├── ms2
|   ├── sync_data
|   |  ├── _2021-08-06-11-23-45
|   |  |  ├── calib.npy
|   |  |  ├── rgb
|   |  |  |  ├── img_left
|   |  |  |  ├── img_right
|   |  ├── _2021-08-13-16-14-48
|   |  ├── ...
|   ├── proj_depth
|   |  ├── _2021-08-06-11-23-45
|   |  |  ├── depth
|   |  |  ├── depth_filtered
|   |  ├── _2021-08-13-16-14-48
|   |  ├── ...
```

## 🙏 Acknowledgments

This project is based on [IGEV](https://github.com/gangweix/IGEV) for the training and evaluation framework, and is inspired by the volume-based depth completion idea in [VPNet](https://ieeexplore.ieee.org/abstract/document/9385917/). We thank the original authors for their excellent works.
