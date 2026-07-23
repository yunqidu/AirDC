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
                    lidar_lines=64,P=None,baseline=None):
    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
    if args.crop_type == 'random':
        transforms_list = [
            transforms.RandomCrop((crop_height, crop_width))
        ]
    elif args.crop_type == 'bottom':
        transforms_list = [
            transforms.BottomCrop((crop_height, crop_width))
        ]
    transform_crop = transforms.Compose(transforms_list)
    target,crop_info = transform_crop(target)
    sparse = apply_crop(sparse,crop_info)

    # 64-line crop.
    keep_ratio = (lidar_lines / 64.0)
    assert keep_ratio >= 0 and keep_ratio <= 1.0, "keep_ratio should be in [0,1]"
    if (keep_ratio < 1.0):
        Km = P[:3,:3].copy()
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
                  P=None,baseline=None):
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
                  crop_width=1216,crop_height=256, lidar_lines=64, P=None, baseline=None):
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


class MS2Depth(data.Dataset):
    """A data loader for the MS2 dataset
    """
    def __init__(self, args,split, howtoval="select",return_left_right=False):
        self.args = args
        self.split = split
        self.howtoval=howtoval
        self.data_folder = args.data_folder
        self.return_left_right=return_left_right
        paths, transform, calib_dict = self.get_paths_and_transform(split) # train or val
        self.paths = paths
        self.transform = transform
        self.calib_dict = calib_dict


    def get_paths_and_transform(self, split):
        import os.path as osp
        root = self.data_folder
        train_seq_list = ['_2021-08-06-11-23-45',  # urban
                          '_2021-08-13-16-14-48',  # Residential
                          '_2021-08-13-16-31-10',  # road1
                          '_2021-08-13-17-06-04',  # campus
                          ]
        test_seq_list = ['_2021-08-13-16-08-46',  # road3
                         ]

        if split == "train":
            use_seqs = train_seq_list
            transform = train_transform
        else: # val, test_completion, test_prediction
            use_seqs = test_seq_list
            if split == "val":
                transform = val_transform
            else:
                transform = test_transform

        img_left_list = []
        img_right_list = []
        disp_gt_list = []
        sparse_hint_list = []
        calib_dict = {}

        for seq in use_seqs:
            seq_imgs = sorted(glob.glob(osp.join(root, f'sync_data/{seq}/rgb/img_left/*.png')))
            if split == 'train':
                seq_imgs = seq_imgs[::3]
            else:
                seq_imgs = seq_imgs[::2]
            
            img_left_list += seq_imgs
            calib_path = osp.join(root, 'sync_data', seq, 'calib.npy')
            if os.path.exists(calib_path):
                calib_dict[seq] = np.load(calib_path, allow_pickle=True).item()

        for samp in img_left_list:
            img_right_list.append(samp.replace('img_left', 'img_right'))
            disp_gt_list.append(
                samp.replace('sync_data', 'proj_depth').replace('img_left', 'depth_filtered'))
            sparse_hint_list.append(
                samp.replace('sync_data', 'proj_depth').replace('img_left', 'depth'))

        # Disable this block when all 1272 frames are required.
        # if split == 'val':
        #     state = np.random.get_state()
        #     np.random.seed(1000)
        #     val_idxs = set(np.random.permutation(len(img_left_list))[:300])
        #     np.random.set_state(state)
        #     
        #     img_left_list = [img_left_list[i] for i in range(len(img_left_list)) if i in val_idxs]
        #     img_right_list = [img_right_list[i] for i in range(len(img_right_list)) if i in val_idxs]
        #     disp_gt_list = [disp_gt_list[i] for i in range(len(disp_gt_list)) if i in val_idxs]
        #     sparse_hint_list = [sparse_hint_list[i] for i in range(len(sparse_hint_list)) if i in val_idxs]
            
        if split == 'train':
            # Optional debug subset that limits the training set to the first 10 samples.
            img_left_list = img_left_list#[:20]
            img_right_list = img_right_list#[:20]
            disp_gt_list = disp_gt_list#[:20]
            sparse_hint_list = sparse_hint_list#[:20]

        if self.return_left_right:
            paths = {
                "left_rgb": img_left_list,
                "right_rgb": img_right_list,
                "d": sparse_hint_list,
                "gt": disp_gt_list,
            }
        else:
            paths = {"rgb": img_left_list, "d": sparse_hint_list, "gt": disp_gt_list}

        return paths, transform, calib_dict

    def __getraw__(self, index):
        if self.return_left_right:
            left_rgb_path = self.paths['left_rgb'][index]
            right_rgb_path = self.paths['right_rgb'][index]
            depth_path = self.paths['d'][index]
            gt_path = self.paths['gt'][index]
            
            # Extract seq from path
            try:
                seq = left_rgb_path.split('sync_data/')[1].split('/')[0]
                calib = self.calib_dict[seq]
            except KeyError:
                found_seq = None
                for s in self.calib_dict.keys():
                    if s in left_rgb_path:
                        found_seq = s
                        break
                if found_seq is not None:
                    calib = self.calib_dict[found_seq]
                    seq = found_seq
                else:
                    raise KeyError(f"Could not find valid calib data for path: {left_rgb_path}. Available keys: {list(self.calib_dict.keys())}")

            left_rgb = np.array(Image.open(left_rgb_path), dtype='uint8')
            right_rgb = np.array(Image.open(right_rgb_path), dtype='uint8')

            K_left = calib['K_rgbL'].astype(np.float32)
            K_right = calib['K_rgbR'].astype(np.float32)
            R_right = calib['R_rgbR'].astype(np.float32)
            T_right = calib['T_rgbR'].astype(np.float32)
            baseline = abs(T_right[0]) * 0.001
            
            P = np.zeros((3, 4), dtype=np.float32)
            P[:3, :3] = K_left
            
            date = seq
            camera_index = 2
        else:
            rgb_path = self.paths['rgb'][index]
            depth_path = self.paths['d'][index]
            gt_path = self.paths['gt'][index]
            
            try:
                seq = rgb_path.split('sync_data/')[1].split('/')[0]
                calib = self.calib_dict[seq]
            except KeyError:
                found_seq = None
                for s in self.calib_dict.keys():
                    if s in rgb_path:
                        found_seq = s
                        break
                if found_seq is not None:
                    calib = self.calib_dict[found_seq]
                    seq = found_seq
                else:
                    raise KeyError(f"Could not find valid calib data for path: {rgb_path}. Available keys: {list(self.calib_dict.keys())}")
            
            rgb = np.array(Image.open(rgb_path), dtype='uint8')
            
            K = calib['K_rgbL'].astype(np.float32)
            T_right = calib['T_rgbR'].astype(np.float32)
            baseline = abs(T_right[0]) * 0.001
            T = T_right
            
            P = np.zeros((3, 4), dtype=np.float32)
            P[:3, :3] = K
            
            date = seq
            camera_index = 2

        depth_pj_png = np.array(Image.open(depth_path), dtype='int')
        target_png = np.array(Image.open(gt_path), dtype='int')

        sparse = depth_pj_png.astype(float) / 256.0
        sparse = np.expand_dims(sparse, -1)

        target = target_png.astype(float) / 256.0
        target = np.expand_dims(target, -1)

        if self.return_left_right:
            return left_rgb, right_rgb, sparse, target, date, camera_index, K_left, K_right, R_right, T_right, P, baseline
        else:
            return rgb, sparse, target, K, T, P, date, camera_index, baseline


    def __getitem__(self, index):
        if self.return_left_right:
            left_rgb, right_rgb, sparse, target, date,camera_index,K_left, K_right, R_right, T_right,P,baseline = self.__getraw__(index)
            K_left=K_left.copy() if isinstance(K_left, np.ndarray) else K_left
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
                                              P=P, baseline=baseline)
            elif self.split == 'val' or self.split == 'test_completion'or self.split == 'test_prediction':
                return_dict = self.transform(self.args, rgb_left=left_rgb,
                                            rgb_right=right_rgb,
                                            sparse=sparse,
                                            target=target,
                                            position=position,
                                            crop_width=self.args.owidth,
                                            crop_height=self.args.oheight,
                                            lidar_lines=self.args.lidar_lines,
                                            P=P, baseline=baseline)
            points_d, d_index = pix2cam(return_dict['sparse'].squeeze(), P, return_idx=True, crop_info=return_dict['crop_info'])
            points_gt, gt_index = pix2cam(return_dict['target'].squeeze(), P, return_idx=True, crop_info=return_dict['crop_info'])
        else:
            rgb, sparse, target, K, T, P,date,camera_index,baseline = self.__getraw__(index)
            K = K.copy() if isinstance(K, np.ndarray) else K
            T = T.copy() if isinstance(T, np.ndarray) else T
            P = P.copy() if isinstance(P, np.ndarray) else P
            if self.split =='train':
               return_dict= self.transform(self.args,rgb=rgb, sparse=sparse,
                                         target=target,
                                          crop_width=self.args.crop_width,
                                          crop_height=self.args.crop_height,
                                            lidar_lines=self.args.lidar_lines,
                                            P=P, baseline=baseline)

               points_d, d_index = pix2cam(return_dict['sparse'].squeeze(), P, return_idx=True,
                                           crop_info=return_dict['crop_info'])
               points_gt, gt_index = pix2cam(return_dict['target'].squeeze(), P, return_idx=True,
                                             crop_info=return_dict['crop_info'])
            elif self.split =='val':
                return_dict = self.transform(self.args,rgb=rgb, sparse=sparse, target=target,
                                                crop_width=self.args.owidth,
                                                  crop_height=self.args.oheight,
                                                 lidar_lines=self.args.lidar_lines,
                                                    P=P, baseline=baseline)
                points_d, d_index = pix2cam(return_dict['sparse'].squeeze(), P, return_idx=True, crop_info=return_dict['crop_info'])
                points_gt, gt_index = pix2cam(return_dict['target'].squeeze(), P, return_idx=True, crop_info=return_dict['crop_info'])
            elif self.split =='test_completion':
                return_dict = self.transform(self.args,rgb=rgb, sparse=sparse, target=target,
                                                  crop_width=self.args.owidth,
                                                  crop_height=self.args.oheight, P=P, baseline=baseline)
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
                'position': return_dict.get('position', position) if 'position' in return_dict else None,
                'crop_top': return_dict['crop_info']['crop_top'],
                'crop_left': return_dict['crop_info']['crop_left'],
            })
            if 'P_crop' in return_dict.keys():
                candidates.update({'P_crop': return_dict['P_crop']})
                
        if 'conversion_rate' in return_dict.keys():
            candidates.update({'conversion_rate': return_dict['conversion_rate']})

        items = {
            key: to_float_tensor(val) if isinstance(val, np.ndarray) and val is not None else val
            for key, val in candidates.items()
        }
        return items


    def __len__(self):
        return len(self.paths['d'])

