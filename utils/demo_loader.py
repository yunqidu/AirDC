import os
import glob
import numpy as np
import torch
from PIL import Image

def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            if ':' not in line: continue
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def depth_read(filename):
    """
    Kitti depth read function.
    """
    depth_png = np.array(Image.open(filename), dtype=int)
    assert(np.max(depth_png) > 255)
    depth = depth_png.astype(float) / 256.
    depth[depth_png == 0] = -1.
    return depth

def read_single_demo_folder(folder_path, args, device):
    """
    Read files from a single demo folder and return a properly formatted dictionary.
    Looks for:
    - *calib_cam_to_cam.txt
    - *gt.png (or just depth)
    - *sparse.png
    """
    calib_files = glob.glob(os.path.join(folder_path, '*calib_cam_to_cam.txt'))
    gt_files = glob.glob(os.path.join(folder_path, '*gt.png'))
    sparse_files = glob.glob(os.path.join(folder_path, '*sparse.png'))
    
    if not calib_files or not sparse_files:
        print(f"[Warning] Cannot find calib or sparse png files in {folder_path}. Generating dummy data fallback.")
        target_h, target_w = args.oheight, args.owidth
        # fallback dummy data
        batch_data = {
            "left_rgb": torch.rand(1, 3, target_h, target_w, device=device),
            "right_rgb": torch.rand(1, 3, target_h, target_w, device=device),
            "d": torch.rand(1, 1, target_h, target_w, device=device),
            "gt": torch.rand(1, 1, target_h, target_w, device=device),
            "P": torch.rand(1, 3, 4, device=device),
            "points_d": np.zeros((1, 3)),
            "d_index": np.zeros((1, 2)),
            "points_gt": np.zeros((1, 3)),
            "gt_index": np.zeros((1, 2)),
            "baseline": torch.tensor([0.54], dtype=torch.float32, device=device)
        }
        return batch_data
        
    calib = read_calib_file(calib_files[0])
    
    # Apply preprocessing consistent with KittiLoader.
    K_left = calib['P_rect_02'].reshape(3, 4)[:, :3].copy()
    
    sparse_depth = depth_read(sparse_files[0])
    h_img, w_img = sparse_depth.shape
    
    # Crop height to target (args.oheight) and width (args.owidth)
    # Apply the default bottom crop used by KittiLoader.
    if hasattr(args, "oheight") and hasattr(args, "owidth"):
        target_h, target_w = getattr(args, "oheight", 256), getattr(args, "owidth", 1216)
        y_start = h_img - target_h if h_img > target_h else 0
        x_start = (w_img - target_w) // 2 if w_img > target_w else 0
        
        sparse_depth = sparse_depth[y_start:y_start+target_h, x_start:x_start+target_w]
        K_left[0, 2] -= x_start
        K_left[1, 2] -= y_start
        
        # Extract point clouds from sparse
        h_crop, w_crop = sparse_depth.shape
        v, u = np.meshgrid(np.arange(h_crop), np.arange(w_crop), indexing='ij')
        valid_mask = sparse_depth > 0
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = sparse_depth[valid_mask]
        
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy
        x_valid = (u_valid - K_left[0, 2]) * z_valid / K_left[0, 0]
        y_valid = (v_valid - K_left[1, 2]) * z_valid / K_left[1, 1]
        
        points_d = np.stack([x_valid, y_valid, z_valid], axis=-1)
        d_index = np.stack([v_valid, u_valid], axis=-1)
        
        if gt_files:
            gt_depth = depth_read(gt_files[0])[y_start:y_start+target_h, x_start:x_start+target_w]
        else:
            gt_depth = np.zeros_like(sparse_depth)
    else:
        points_d = np.zeros((1, 3))
        d_index = np.zeros((1, 2))
        gt_depth = np.zeros_like(sparse_depth)

    P = calib['P_rect_02'].reshape(3, 4).copy()
    P[0, 2] -= x_start
    P[1, 2] -= y_start

    # Build the returned batch dictionary with tensor conversion and a batch dimension.
    batch_data = {
        "left_rgb": torch.rand(1, 3, target_h, target_w, device=device),  # Dummy if no RGB found
        "right_rgb": torch.rand(1, 3, target_h, target_w, device=device), # Dummy if no right RGB
        "d": torch.tensor(sparse_depth, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0), # (1,1,H,W)
        "gt": torch.tensor(gt_depth, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0),
        "P": torch.tensor(P, dtype=torch.float32, device=device).unsqueeze(0),
        "points_d": points_d[np.newaxis, ...],
        "d_index": d_index[np.newaxis, ...],
        "points_gt": points_d[np.newaxis, ...], # dummy for gt points
        "gt_index": d_index[np.newaxis, ...],   # dummy
        "baseline": torch.tensor([0.54], dtype=torch.float32, device=device)
    }

    return batch_data
