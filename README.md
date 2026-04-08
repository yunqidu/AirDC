<h2>
<a href="https://github.com/duyunqi/AirDC/" target="_blank">AirDC: Adaptive Iterative Depth Refinement Framework for Full-Range Absolute Depth Completion</a>
</h2>
    
This is the official PyTorch implementation of "AirDC: Adaptive Iterative Depth Refinement Framework for Full-Range Absolute Depth Completion"

<img width="431" height="304" alt="gpu" src="https://github.com/user-attachments/assets/766b7305-0045-496c-b2a4-1d4572719a39"/>


## 🔍 Introduction
 



## 💻 Code 
coming soon...


## 💾 Datasets
We used two datasets(KITTIDC and Virtual KITTI 2.0) for training and three datasets for evaluation.
### KITTI Depth Completion (KITTI DC)

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
```
├── vkitti2
|   ├── Scene01
|   |  ├── 15-deg-left
|   |  |  ├──── frames
|   |  |  |  ├── depth
|   |  |  |  ├── rgb      
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
