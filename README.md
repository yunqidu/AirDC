<h2>
<a href="https://github.com/duyunqi/AirDC/" target="_blank">RefDC : Iterative Hypothesis-guided Depth Refinement for Long Range Stereo-LiDAR Depth Completion </a>
</h2>
    
This is the official PyTorch implementation of "AirDC: Adaptive Iterative Depth Refinement Framework for Full-Range Absolute Depth Completion"

![git photo](https://github.com/user-attachments/assets/60a395bc-f32a-4488-84ed-d1664f3662c1)


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

