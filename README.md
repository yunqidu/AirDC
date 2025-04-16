RefDC:Iterative Hypothesis-guided Depth Refinement for Long Range Stereo-LiDAR Depth Completion
This is the official PyTorch implementation of "RefDC:Iterative Hypothesis-guided Depth Refinement for Long Range Stereo-LiDAR Depth Completion"
![Comp](https://github.com/user-attachments/assets/ffb6b265-f8f4-4042-a48b-6a8161ee54c3)![Comp](https://github.com/user-attachments/assets/10e7c20e-f01d-4e3a-be9d-1cad3ed3b023)# 
## Introduction
Depth completion aims to generate dense and highly accurate depth information from sparse measurements, which is essential for robotic perception and autonomous driving systems. Although existing stereo-LiDAR-based methods have shown promising accuracy, they encounter limitations in long-range depth estimation due to errors introduced during the disparity-depth conversion process and accumulated in the propagation of pyramid features. Furthermore, these methods suffer from high computational costs associated with 3D convolutions. To overcome these issues, we introduce RefDC, an iterative hypothesis-guided depth refinement framework for long-range-focused depth completion, which offers two key contributions: 1) We introduce a novel SLDV **S**tereo-**L**iDAR **D**epth **V**olume) that constructs the volumetric representation using true metric depth values as the depth dimension, enabling more accurate and physically consistent depth modeling. Additionally, it interpolates the point cloud into neighboring voxels during downsampling, which effectively reduces errors arising from excessive reliance on disparity-based conversion; 2) We propose an IHDR (**I**terative **H**ypothesis-guided **D**epth **R**efinement) module that extracts multi-scale features from various depth hypotheses and performs ConvGRU iterations, which progressively refine depth estimates, particularly those at long distances. RefDC significantly alleviates the computational load and mitigates the over-smoothing problem encountered in 3D convolutions. Experimental evaluations on both real-world and synthetic outdoor datasets demonstrate that our method achieves state-of-the-art accuracy (4.23% and 11.9% improvement respectivly) and robustness, particularly excelling in long-range depth estimation (a 6% increase in performance). 


## Code 
coming soon...


## Datasets
We used two datasets(KITTIDC and Virtual KITTI 2.0) for training and three datasets for evaluation.
### KITTI Depth Completion (KITTI DC)
#### KITTI Depth Completion (KITTI DC)

KITTI DC dataset is available at the [KITTI DC Website](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion).
For color images, KITTI Raw dataset is also needed, which is available at the [KITTI Raw Website](http://www.cvlibs.net/datasets/kitti/raw_data.php). 

The overall data directory is structured as follows:

```
├── kitti_depth
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
├── vkitti2
|   ├── Scene01
|   |  ├── 15-deg-left
|   |  |  ├──── frames
|   |  |  |  ├── depth
|   |  |  |  ├── rgb      
