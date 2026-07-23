# !/usr/bin/env python -u
import os
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import utils.transforms as transforms
from utils.basic import pix2cam

import glob
from random import sample
import torchvision

howtoval = 'full'
jitter = 0.1  # color jitter for images
to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()

import numpy as np


class AddCoordsNp():
    """Add coords to a tensor"""

    def __init__(self, x_dim=64, y_dim=64, with_r=False):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r

    def call(self):
        """
        input_tensor: (batch, x_dim, y_dim, c)
        """
        xx_ones = np.ones([self.x_dim], dtype=np.int32)
        xx_ones = np.expand_dims(xx_ones, 1)

        xx_range = np.expand_dims(np.arange(self.y_dim), 0)

        xx_channel = np.matmul(xx_ones, xx_range)
        xx_channel = np.expand_dims(xx_channel, -1)

        yy_ones = np.ones([self.y_dim], dtype=np.int32)
        yy_ones = np.expand_dims(yy_ones, 0)

        yy_range = np.expand_dims(np.arange(self.x_dim), 1)

        yy_channel = np.matmul(yy_range, yy_ones)
        yy_channel = np.expand_dims(yy_channel, -1)

        xx_channel = xx_channel.astype('float32') / (self.y_dim - 1)
        yy_channel = yy_channel.astype('float32') / (self.x_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        ret = np.concatenate([xx_channel, yy_channel], axis=-1)

        if self.with_r:
            rr = np.sqrt(np.square(xx_channel - 0.5) + np.square(yy_channel - 0.5))
            ret = np.concatenate([ret, rr], axis=-1)

        return ret

def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data
def get_params(calib_dir, cam_index):
    if calib_dir.endswith('_'):
        cam2cam = read_calib_file(calib_dir + 'calib_cam_to_cam.txt')
        velo2cam = read_calib_file(calib_dir + 'calib_velo_to_cam.txt')
    else:
        cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
        velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))

    # Intrinsic matrix K.
    K_key = f'K_{cam_index:02d}'
    K_params = cam2cam[K_key].reshape(3, 3)
    # Distortion coefficients D.
    D_key = f'D_{cam_index:02d}'
    D_params = cam2cam[D_key]

    R_cam=f"R_{cam_index:02d}"
    T_cam=f"T_{cam_index:02d}"
    R_cam_params = cam2cam[R_cam].reshape(3, 3)
    T_cam_params = cam2cam[T_cam].reshape(3, 1)

    # Rectification rotation matrix R_rect.
    R_key = f'R_rect_{cam_index:02d}'
    R_rect_params = cam2cam[R_key].reshape(3, 3)
    # Build the 4x4 rectification matrix.
    R_rect_4x4 = np.eye(4)
    R_rect_4x4[:3, :3] = R_rect_params

    # Build the 4x4 LiDAR-to-camera transformation matrix.
    RT_velo_to_cam = np.eye(4)
    # Read LiDAR-to-camera rotation and translation from velo2cam.
    R_velo_to_cam = velo2cam['R'].reshape(3, 3)
    T_velo_to_cam = velo2cam['T'].reshape(3, 1)

    RT_velo_to_cam[:3, :3] = R_velo_to_cam  # 3x3 rotation matrix from calib_velo_to_cam.
    RT_velo_to_cam[:3, 3] = T_velo_to_cam.flatten()  # 3x1 translation vector from calib_velo_to_cam.

    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0' + str(cam_index)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
    K_params = P_rect[:3, :3]
    return K_params, D_params,R_cam_params,T_cam_params, RT_velo_to_cam, P_rect
def load_all_calib(calib_dir):
    calib_params = {}

    for calib_file in os.listdir(calib_dir):
        if calib_file.endswith("_calib_cam_to_cam.txt"):
            date = "_".join(calib_file.split('_')[:3])
            path_calib = os.path.join(calib_dir, date + "_")
            calib_params[date] = {}
            for cam_index in [2, 3]:
                K_params, D_params,R_cam_params,T_cam_params, RT_velo_to_cam, P_velo2im = get_params(path_calib, cam_index)
                calib_params[date][cam_index] = {'K': K_params, 'D': D_params,
                                                 'R': R_cam_params,'T': T_cam_params,
                                                 'RT': RT_velo_to_cam, 'P': P_velo2im}
    return calib_params


def normalize_rgb(rgb, mean, std):
    # Normalize RGB values in the NumPy array to [0, 1].
    rgb = rgb / 255.0  # Normalize pixel values from [0, 255] to [0, 1].

    # Standardize with mean and standard deviation.
    mean = np.array(mean).reshape((1, 1, 3))
    std = np.array(std).reshape((1, 1, 3))

    # Apply channel-wise standardization.
    rgb = (rgb - mean) / std
    return rgb


def apply_crop(data,crop_info):
    if data is None:
        return None, None
    top = crop_info['crop_top']
    left = crop_info['crop_left']
    h = crop_info['crop_height']
    w = crop_info['crop_width']

    if data.ndim == 3:
        return data[top:top + h, left:left + w, :]
    elif data.ndim == 2:
        return data[top:top + h, left:left + w]
    else:
        raise RuntimeError("Invalid data dimension")

def train_transform(args,rgb=None, rgb_left=None, rgb_right=None,
                    sparse=None, target=None,position=None,crop_width=512,crop_height=256,
                    lidar_lines=64,P=None):
    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
    if args.crop_type == 'random':
        transforms_list = [
            transforms.RandomCrop((crop_height, crop_width))
        ]
    elif args.crop_type == 'bottom':
        transforms_list = [
            transforms.BottomCrop((crop_height, crop_width))
        ]
    # Apply the configured spatial crop.
    transform_crop = transforms.Compose(transforms_list)
    target,crop_info = transform_crop(target)
    sparse = apply_crop(sparse,crop_info)

    # Simulate lower-line LiDAR sampling when requested.
    keep_ratio = (lidar_lines / 64.0)
    assert keep_ratio >= 0 and keep_ratio <= 1.0, "keep_ratio should be in [0,1]"
    if (keep_ratio < 1.0):
        Km =P[:3,:3].copy()
        Km[0, 2] -= crop_info['crop_left']  # cx' = cx_original - crop_left
        Km[1, 2] -= crop_info['crop_top']  # cy' = cy_original - crop_top
        sparse = sample_lidar_lines(
            sparse,
            intrinsics=Km,  # Adjusted intrinsics.
            keep_ratio=keep_ratio
        )

    brightness = np.random.uniform(max(0, 1 - jitter),
                                   1 + jitter)
    contrast = np.random.uniform(max(0, 1 - jitter), 1 + jitter)
    saturation = np.random.uniform(max(0, 1 - jitter),
                                   1 + jitter)
    transform_rgb = transforms.Compose([
        transforms.ColorJitter(brightness, contrast, saturation, 0)
    ])
    if rgb_left is not None:
        rgb_left = transform_rgb(rgb_left)
        rgb_left=apply_crop(rgb_left,crop_info)
        rgb_left = normalize_rgb(rgb_left, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if rgb_right is not None:
        rgb_right = transform_rgb(rgb_right)
        rgb_right=apply_crop(rgb_right,crop_info)
        rgb_right = normalize_rgb(rgb_right, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if rgb is not None:
        rgb = transform_rgb(rgb)
        rgb=apply_crop(rgb,crop_info)
        rgb_norm = normalize_rgb(rgb, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if position is not None:
        position = apply_crop(position,crop_info)

    if rgb_right is not None:
        return {
            'rgb_left': rgb_left,
            'rgb_right': rgb_right,
            'sparse': sparse,
            'target': target,
            'position': position,
            'crop_info': crop_info
        }
    else:
        return {
            'rgb': rgb,
            'rgb_norm': rgb_norm,
            'sparse': sparse,
            'target': target,
            'position': position,
            'crop_info': crop_info
        }


def val_transform(args,rgb=None,rgb_left=None, rgb_right=None,
                  sparse=None, target=None,position=None,
                  crop_width=1216,crop_height=352,lidar_lines=64,
                  P=None):
    transform = transforms.Compose([
        transforms.BottomCrop((crop_height, crop_width)),
    ])
    if rgb_left is not None:
        rgb_left,crop_info = transform(rgb_left)
        rgb_left=normalize_rgb(rgb_left, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if rgb_right is not None:
        rgb_right,_ = transform(rgb_right)
        rgb_right=normalize_rgb(rgb_right, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if rgb is not None:
        rgb,crop_info = transform(rgb)
        rgb_norm = normalize_rgb(rgb, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if sparse is not None:
        sparse,_ = transform(sparse)
        # 64-line crop.
        keep_ratio = (lidar_lines / 64.0)
        assert keep_ratio >= 0 and keep_ratio <= 1.0, "keep_ratio should be in [0,1]"
        if (keep_ratio < 1.0):
            Km = P[:3, :3].copy()
            Km[0, 2] -= crop_info['crop_left']  # cx' = cx_original - crop_left
            Km[1, 2] -= crop_info['crop_top']  # cy' = cy_original - crop_top
            sparse = sample_lidar_lines(
                sparse,
                intrinsics=Km,  # Adjusted intrinsics.
                keep_ratio=keep_ratio
            )

    if target is not None:
        target,_ = transform(target)
    if position is not None:
        position,_ = transform(position)
    if P is not None:
        P_crop = P.copy()
        P_crop[0, 2] -= crop_info['crop_left']  # cx' = cx_original - crop_left
        P_crop[1, 2] -= crop_info['crop_top']  # cy' = cy_original - crop_top
    conversion_rate = None

    if rgb_right is not None:
        result = {
            'rgb_left':rgb_left,
            'rgb_right':rgb_right,
           'sparse':sparse,
            'target':target,
            'position':position,
            'P_crop':P_crop,
            'crop_info':crop_info
        }
        return result
    else:
        result = {
            'rgb':rgb,
            'rgb_norm':rgb_norm,
           'sparse':sparse,
            'target':target,
            'position':position,
            'P_crop':P_crop,
            'crop_info':crop_info
        }
        return result



def test_transform(args,rgb=None,rgb_left=None, rgb_right=None,
                  sparse=None, target=None,position=None,
                  crop_width=1216,crop_height=256):
    transform = transforms.Compose([
        transforms.BottomCrop((crop_height, crop_width)),
    ])
    if rgb_left is not None:
        rgb_left,crop_info = transform(rgb_left)
        if args.rgb_norm:
            rgb_left=normalize_rgb(rgb_left, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if rgb_right is not None:
        rgb_right,_ = transform(rgb_right)
        if args.rgb_norm:
            rgb_right=normalize_rgb(rgb_right, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if rgb is not None:
        rgb,crop_info = transform(rgb)
        if args.rgb_norm:
            rgb_norm = normalize_rgb(rgb, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if sparse is not None:
        sparse,_ = transform(sparse)
        # 64-line crop.
    if target is not None:
        target,_ = transform(target)
    if position is not None:
        position,_ = transform(position)
    if rgb_right is not None:
        return {
            'rgb_left':rgb_left,
            'rgb_right':rgb_right,
            'sparse':sparse,
            'target':target,
            'position':position,
            'crop_info':crop_info
        }
    else:
        return {
            'rgb': rgb,
            'rgb_norm': rgb_norm,
            'sparse': sparse,
            'target': target,
            'position': position,
            'crop_info': crop_info
        }


def sample_lidar_lines(
    depth_map: np.ndarray, intrinsics: np.ndarray, keep_ratio: float = 1.0
) -> np.ndarray:
    """
    Takes in input a depth map generated by a 64 line lidar and sparsify the number of
    lines used, returning a sparse depth map with less lidar lines.
    Parameters
    ----------
    depth_map: array like
        sparse depth map of shape H x W x 1
    intrinsics: array like
        the intrinsic parameters of shape 3 x 3
    keep_ratio: float, default 1.0
        the sparsification parameter, 1.0 is 64 lines, 0.50 roughly 32 lines and so on.
    Returns
    -------
    sparse_depth_map: array like
        the sparsified depth map of shape H x W x 1
    """

    v, u,_ = np.nonzero(depth_map)
    z = depth_map[v, u, 0]
    points = np.linalg.inv(intrinsics) @ (np.vstack([u, v, np.ones_like(u)]) * z)
    points = points.transpose([1, 0])

    scan_y = points[:, 1]
    distance = np.linalg.norm(points, 2, axis=1)
    pitch = np.arcsin(scan_y / distance)
    num_points = np.shape(pitch)[0]
    pitch = np.reshape(pitch, (num_points, 1))

    max_pitch = np.max(pitch)
    min_pitch = np.min(pitch)
    angle_interval = (max_pitch - min_pitch) / 64.0
    angle_label = np.round((pitch - min_pitch) / angle_interval)
    sampling_mask = angle_label % (1.0 / keep_ratio) == 0

    final_mask = np.zeros_like(depth_map, dtype=bool)
    final_mask[depth_map[..., 0] > 0] = sampling_mask
    sampled_depth = np.zeros_like(final_mask, dtype=np.float32)
    sampled_depth[final_mask] = depth_map[final_mask]
    return sampled_depth


class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset
    """
    def __init__(self, args,split, howtoval="select",return_left_right=False):
        self.args = args
        self.split = split
        self.howtoval=howtoval
        self.data_folder = args.data_folder
        self.return_left_right=return_left_right
        paths, transform = self.get_paths_and_transform(split) # train or val
        self.paths = paths
        self.transform = transform
        self.threshold_translation = 0.1
        self.calib_params = load_all_calib(args.calib_folder)


    def get_paths_and_transform(self, split):
        def get_rgb_paths(p):
            normalized_path = os.path.normpath(p)
            parts = normalized_path.split(os.sep)

            try:
                # Find the KITTI_Depth_Completion path component.
                base_idx = parts.index("KITTI_Depth_Completion")
            except ValueError:
                raise ValueError("Required path component KITTI_Depth_Completion does not exist")

            # Compute the target position relative to base_idx.
            target_idx = base_idx + 3
            if target_idx >= len(parts):
                raise ValueError("Unexpected path structure")
                # Extract the date component in YYYY_MM_DD format.
            try:
                date_drive_id =  parts[target_idx]
            except IndexError:
                raise IndexError("Path index out of range")
            date = date_drive_id.split('_drive_')[0]
            return os.path.join(self.data_folder, 'raw', date, date_drive_id, parts[-2], 'data', parts[-1])

        if split == "train":
            transform = train_transform
            if self.return_left_right:
                glob_d = os.path.join(self.data_folder,
                                      'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_02/*.png')
                glob_gt = os.path.join(self.data_folder, 'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_02/*.png')
            else:
                glob_d = os.path.join(self.data_folder, 'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png')
                glob_gt = os.path.join(self.data_folder, 'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png')
            if glob_gt is not None:
                paths_d = sorted(glob.glob(glob_d))
                paths_gt = sorted(glob.glob(glob_gt))
                paths_rgb = sorted([get_rgb_paths(p) for p in paths_d])

        elif split == "val":
            transform = val_transform
            if self.howtoval=="full":
                if self.return_left_right:
                    glob_gt = os.path.join(self.data_folder,
                                           'data_depth_annotated/val/*_sync/proj_depth/groundtruth/image_02/*.png')
                else:
                    glob_d = os.path.join(self.data_folder,
                                          'data_depth_velodyne/val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png')
                    glob_gt = os.path.join(self.data_folder,
                                           'data_depth_annotated/val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png')
                if glob_gt is not None:
                    paths_gt = sorted(glob.glob(glob_gt))
                    paths_d=[path.replace("groundtruth","velodyne_raw").replace("data_depth_annotated","data_depth_velodyne") for path in paths_gt]
                    paths_rgb = sorted([get_rgb_paths(p) for p in paths_d])
            elif self.howtoval=="select":
                glob_d = os.path.join(self.data_folder, "data_depth_selection/val_selection_cropped/velodyne_raw/*.png")
                glob_gt = os.path.join(self.data_folder,
                                       "data_depth_selection/val_selection_cropped/groundtruth_depth/*.png")
                glob_rgb = os.path.join(self.data_folder, "data_depth_selection/val_selection_cropped/image/*.png")

                if glob_gt is not None:
                    paths_d = sorted(glob.glob(glob_d))
                    paths_gt = sorted(glob.glob(glob_gt))
                    paths_rgb = sorted(glob.glob(glob_rgb))
        elif split == "test_completion":
            transform = test_transform
            glob_d = os.path.join(
                self.data_folder,
                "data_depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png"
            )
            glob_gt = None  # "test_depth_completion_anonymous/"
            glob_rgb = os.path.join(
                self.data_folder,
                "data_depth_selection/test_depth_completion_anonymous/image/*.png")
            if glob_gt is not None:
                paths_d = sorted(glob.glob(glob_d))
                paths_gt = sorted(glob.glob(glob_gt))
                paths_rgb = sorted([get_rgb_paths(p) for p in paths_d])
            else:
                # Test split does not provide ground-truth depth.
                paths_rgb = sorted(glob.glob(glob_rgb))
                paths_gt = [None] * len(paths_rgb)
                if split == "test_prediction":
                    paths_d = [None] * len(
                        paths_rgb)  # Depth prediction test split has no sparse depth.
                else:
                    paths_d = sorted(glob.glob(glob_d))
        elif split == "test_prediction":
            transform = test_transform
            glob_d = None
            glob_gt = None  # "test_depth_completion_anonymous/"
            glob_rgb = os.path.join(
                self.data_folder,
                "data_depth_selection/test_depth_prediction_anonymous/image/*.png")
            if glob_gt is not None:
                paths_d = sorted(glob.glob(glob_d))
                paths_gt = sorted(glob.glob(glob_gt))
                paths_rgb = sorted([get_rgb_paths(p) for p in paths_d])
            else:
                # Test split does not provide ground-truth depth.
                paths_rgb = sorted(glob.glob(glob_rgb))
                paths_gt = [None] * len(paths_rgb)
                if split == "test_prediction":
                    paths_d = [None] * len(
                        paths_rgb)  # Depth prediction test split has no sparse depth.
                else:
                    paths_d = sorted(glob.glob(glob_d))
        else:
            raise ValueError("Unrecognized split " + str(split))
        if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
            print("Length of paths_rgb, paths_d, paths_gt not equal!",len(paths_rgb), len(paths_d), len(paths_gt))
        if self.return_left_right:
            # Split stereo image paths by camera suffix.
            paths_left = paths_rgb
            paths_right = [path.replace('image_02', 'image_03') for path in paths_left]
            paths = {
                "left_rgb": paths_left,
                "right_rgb": paths_right,
                "d": paths_d,
                "gt": paths_gt,
            }
        else:
            paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}

        return paths, transform

    def sample_paths(self, paths, sample_size):
        if sample_size == 0:
            # Return empty lists for all keys.
            return {key: [] for key in paths.keys()}
        else:
            # Get the total length assuming all keys have equal-length values.
            total_length = len(next(iter(paths.values())))  # Length of the first key.
            sample_indices = random.sample(range(total_length), sample_size)  # Generate sampled indices.

            # Sample all dictionary values by index.
            sampled_paths = {
                key: [value[i] for i in sample_indices] for key, value in paths.items()
            }

        return sampled_paths


    def __getraw__(self, index):
        if self.return_left_right:
            left_rgb_path = self.paths['left_rgb'][index]
            right_rgb_path = self.paths['right_rgb'][index]
            if self.split == "val" and self.howtoval == "select":
                date = os.path.basename(left_rgb_path)[:10]
                camera_index = int(os.path.basename(left_rgb_path).split('image_0')[-1][0])
            else:
                normalized_path = os.path.normpath(left_rgb_path)
                parts = normalized_path.split(os.sep)
                date_idx = parts.index("KITTI_Depth_Completion")+3
                date = parts[date_idx].split('_drive_')[0]

                camera_index = int(left_rgb_path.split('image_0')[1][0])

            calib_left = self.calib_params[date][2]
            calib_right = self.calib_params[date][3]
            left_rgb = np.array(Image.open(left_rgb_path), dtype='uint8')
            right_rgb = np.array(Image.open(right_rgb_path), dtype='uint8')

            depth_path = self.paths['d'][index]
            gt_path = self.paths['gt'][index]
            K_left = calib_left['K']
            K_right = calib_right['K']
            R_right = calib_right['R']
            T_right = calib_right['T']
            P = calib_left['P']

        else:
            rgb_path = self.paths['rgb'][index]
            if self.split == "val" and self.howtoval == "select":
                date = os.path.basename(rgb_path)[:10]
                camera_index = int(os.path.basename(rgb_path).split('image_0')[-1][0])
            else:
                date = rgb_path.split('/')[5]
                camera_index = int(rgb_path.split('image_0')[1][0])
            calib = self.calib_params[date][camera_index]
            rgb = np.array(Image.open(rgb_path), dtype='uint8')

            depth_path = self.paths['d'][index]
            gt_path = self.paths['gt'][index]
            K = calib['K']
            T = calib['RT'][:3, 3]
            P = calib['P']

        # Read KITTI depth maps from 16-bit PNG files.
        depth_pj_png = np.array(Image.open(depth_path), dtype='int')
        target_png = np.array(Image.open(gt_path), dtype='int')

        assert np.max(depth_pj_png) > 255 and np.max(depth_pj_png), "np.max(depth_pj_png)={}".format(
            np.max(depth_pj_png))  # Validate the expected 16-bit depth encoding.


        sparse = depth_pj_png.astype(float) / 256
        sparse = np.expand_dims(sparse, -1)  # (375, 1242, 1)->(1, 375, 1242)

        target = target_png.astype(float) / 256
        target = np.expand_dims(target, -1)  # (375, 1242, 1)->(1, 375, 1242)

        if self.return_left_right:
            return left_rgb, right_rgb, sparse, target, date,camera_index,K_left, K_right, R_right, T_right,P
        else:
            return rgb, sparse, target, K, T, P,date,camera_index

    def __getitem__(self, index):
        if self.return_left_right:
            left_rgb, right_rgb, sparse, target, date,camera_index,K_left, K_right, R_right, T_right,P = self.__getraw__(index)
            K_left=K_left.copy() if isinstance(K_left, np.ndarray) else K_left#Unrectified intrinsics; not used.
            K_right=K_right.copy() if isinstance(K_right, np.ndarray) else K_right
            R_right=R_right.copy() if isinstance(R_right, np.ndarray) else R_right
            T_right=T_right.copy() if isinstance(T_right, np.ndarray) else T_right
            P=P.copy() if isinstance(P, np.ndarray) else P
            
            crop_height = self.args.crop_height if self.split == "train" else self.args.oheight
            crop_width = self.args.crop_width if self.split == "train" else self.args.owidth

            position = AddCoordsNp(left_rgb.shape[0], left_rgb.shape[1])
            position = position.call()
            if self.split == "train":
                return_dict= self.transform(self.args,rgb_left=left_rgb,
                                              rgb_right=right_rgb,
                                              sparse=sparse,
                                              target=target,
                                              position=position,
                                              crop_width=self.args.crop_width,
                                              crop_height=self.args.crop_height,
                                              lidar_lines=self.args.lidar_lines,
                                              P=P)
            elif self.split == 'val' or self.split == 'test_completion'or self.split == 'test_prediction':
                return_dict = self.transform(self.args, rgb_left=left_rgb,
                                            rgb_right=right_rgb,
                                            sparse=sparse,
                                            target=target,
                                            position=position,
                                            crop_width=self.args.owidth,
                                            crop_height=self.args.oheight,
                                            lidar_lines=self.args.lidar_lines,
                                            P=P)
            points_d, d_index = pix2cam(return_dict['sparse'].squeeze(), P, return_idx=True, crop_info=return_dict['crop_info'])
            points_gt, gt_index = pix2cam(return_dict['target'].squeeze(), P, return_idx=True, crop_info=return_dict['crop_info'])
        else:
            rgb, sparse, target, K, T, P,date,camera_index = self.__getraw__(index)
            K = K.copy() if isinstance(K, np.ndarray) else K
            T = T.copy() if isinstance(T, np.ndarray) else T
            P = P.copy() if isinstance(P, np.ndarray) else P
            if self.split =='train':
               return_dict= self.transform(rgb=rgb, sparse=sparse,
                                         target=target,
                                          crop_width=self.args.crop_width,
                                          crop_height=self.args.crop_height,
                                            lidar_lines=self.args.lidar_lines,
                                            P=P)

               points_d, d_index = pix2cam(return_dict['sparse'].squeeze(), P, return_idx=True,
                                           crop_info=return_dict['crop_info'])
               points_gt, gt_index = pix2cam(return_dict['target'].squeeze(), P, return_idx=True,
                                             crop_info=return_dict['crop_info'])
            elif self.split =='val':
                return_dict = self.transform(rgb=rgb, sparse=sparse, target=target,
                                                crop_width=self.args.owidth,
                                                  crop_height=self.args.oheight,
                                                 lidar_lines=self.args.lidar_lines,
                                                    P=P)
                points_d, d_index = pix2cam(return_dict['sparse'].squeeze(), P, return_idx=True, crop_info=return_dict['crop_info'])
                points_gt, gt_index = pix2cam(return_dict['target'].squeeze(), P, return_idx=True, crop_info=return_dict['crop_info'])
            elif self.split =='test_completion':
                return_dict = self.transform(rgb=rgb, sparse=sparse, target=target,
                                                  crop_width=self.args.owidth,
                                                  crop_height=self.args.oheight,)
                points_d, d_index = pix2cam(return_dict['sparse'].squeeze(), P, return_idx=True, crop_info=return_dict['crop_info'])
                points_gt, gt_index = pix2cam(return_dict['target'].squeeze(), P, return_idx=True, crop_info=return_dict['crop_info'])
            else:
                raise NotImplementedError (f"split should be train or val,but got {self.split}")
        candidates = {"d": return_dict['sparse'], "points_d": points_d,
                "d_index": d_index, "gt": return_dict['target'],
                "points_gt": points_gt, "gt_index": gt_index,
                'date': date, 'camera_index': camera_index, }

        if self.return_left_right: # Return crop metadata for left-right validation samples.
            candidates.update({"left_rgb": return_dict['rgb_left'],
                               "right_rgb": return_dict['rgb_right'],
                               'crop_top': return_dict['crop_info']['crop_top'],
                               'crop_left': return_dict['crop_info']['crop_left'],
                               'position':return_dict['position'],
                               'P': P,
                               })
            if 'P_crop' in return_dict.keys():
                candidates.update({'P_crop': return_dict['P_crop']})
        else:
            candidates.update({
                'K': K, 'T': T, 'P': P,"rgb": return_dict['rgb'], "rgb_norm": return_dict['rgb_norm'],
                'position': return_dict.get('position', position),
                'crop_top': return_dict['crop_info']['crop_top'],
                'crop_left': return_dict['crop_info']['crop_left'],
            })
            if 'P_crop' in return_dict.keys():
                candidates.update({'P_crop': return_dict['P_crop']})

        items = {
            key: to_float_tensor(val) if isinstance(val, np.ndarray) and val is not None else val
            for key, val in candidates.items()
        }
        return items

    def __len__(self):
        return len(self.paths['d'])
