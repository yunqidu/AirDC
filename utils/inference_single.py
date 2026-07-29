import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

workspace_path = str(Path(__file__).resolve().parents[1])
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)

from model.AirDC.airdc_model import AirDC
from utils.basic import pix2cam, reshape_to_BHW
from utils.fit_utils import Config, custom_collate_fn_with_batch, load_checkpoint_robust, to_float_tensor
from utils.kitti_loader import val_transform
from utils.visual_loss import visualize_multi, visualize_single


width_to_P = {
    1241: np.array([[7.188560e02, 0.0, 6.071928e02, 0.0],
                    [0.0, 7.188560e02, 1.852157e02, 0.0],
                    [0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
    1242: np.array([[7.215377e02, 0.0, 6.095593e02, 0.0],
                    [0.0, 7.215377e02, 1.728540e02, 0.0],
                    [0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
    1224: np.array([[707.0493, 0.0, 6.095593e02, 0.0],
                    [0.0, 7.215377e02, 1.728540e02, 0.0],
                    [0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
    1226: np.array([[708.2046, 0.0, 6.071928e02, 0.0],
                    [0.0, 7.188560e02, 1.852157e02, 0.0],
                    [0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
    1238: np.array([[718.3351, 0.0, 6.003891e02, 0.0],
                    [0.0, 7.183351e02, 1.815122e02, 0.0],
                    [0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
}


def _resolve_path(path, base):
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else Path(base) / path


def _resolve_config_path(config_name):
    config_path = Path(config_name)
    if config_path.suffix not in {'.yaml', '.yml'}:
        config_path = Path('config') / f'{config_name}.yaml'
    return _resolve_path(config_path, workspace_path)


def _normalize_io_dir(directory, default_subpath):
    path = Path(directory) if directory else Path(workspace_path) / default_subpath
    return str(path if path.is_absolute() else Path(workspace_path) / path)


def _scan_indices(input_dir):
    possible_indices = set()
    for filename in os.listdir(input_dir):
        match = re.match(r'(\d+)', filename)
        if match:
            possible_indices.add(int(match.group(1)))
    required = ['left.png', 'right.png', 'sparse.png', 'gt.png']
    return [idx for idx in sorted(possible_indices)
            if all(os.path.exists(os.path.join(input_dir, f'{idx}{suffix}')) for suffix in required)]


def _load_config(config_name):
    config_path = _resolve_config_path(config_name)
    if not config_path.exists():
        raise FileNotFoundError(f'Configuration file not found: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f) or {}
    config = Config(**config_dict)
    if config.model_name != 'AirDC':
        raise ValueError('inference_single.py supports only AirDC')
    log_dir = Path(workspace_path) / 'log'
    if getattr(config, 'ckpt_path', ''):
        ckpt = config.ckpt_path[0] if isinstance(config.ckpt_path, list) else config.ckpt_path
        config.ckpt_path = str(_resolve_path(ckpt, log_dir))
    return config


def _prepare_sample(args, index, input_dir):
    left_rgb = np.array(Image.open(os.path.join(input_dir, f'{index}left.png')), dtype=np.uint8)
    right_rgb = np.array(Image.open(os.path.join(input_dir, f'{index}right.png')), dtype=np.uint8)
    sparse_png = np.array(Image.open(os.path.join(input_dir, f'{index}sparse.png')), dtype=np.int32)
    gt_png = np.array(Image.open(os.path.join(input_dir, f'{index}gt.png')), dtype=np.int32)

    sparse = np.expand_dims(sparse_png.astype(np.float32) / 256.0, -1)
    target = np.expand_dims(gt_png.astype(np.float32) / 256.0, -1)
    P_np = width_to_P.get(left_rgb.shape[1], width_to_P[1242])
    return_dict = val_transform(
        args,
        rgb_left=left_rgb,
        rgb_right=right_rgb,
        sparse=sparse,
        target=target,
        P=P_np,
        crop_width=args.owidth,
        crop_height=args.oheight,
        lidar_lines=args.lidar_lines,
    )
    points_d, d_index = pix2cam(return_dict['sparse'].squeeze(), P_np, return_idx=True,
                                crop_info=return_dict['crop_info'])
    points_gt, gt_index = pix2cam(return_dict['target'].squeeze(), P_np, return_idx=True,
                                  crop_info=return_dict['crop_info'])
    batch_item = {
        'P': torch.tensor(P_np, dtype=torch.float32),
        'left_rgb': return_dict['rgb_left'],
        'right_rgb': return_dict['rgb_right'],
        'crop_top': return_dict['crop_info']['crop_top'],
        'crop_left': return_dict['crop_info']['crop_left'],
        'd': return_dict['sparse'],
        'points_d': points_d,
        'd_index': d_index,
        'gt': return_dict['target'],
        'points_gt': points_gt,
        'gt_index': gt_index,
    }
    return {key: to_float_tensor(val) if isinstance(val, np.ndarray) and val is not None else val
            for key, val in batch_item.items()}


def inference_single(index_list, config_name='val_all_att', iter_num=None, iter_vis=False,
                     max_batch_size=5, output_dir=None, input_dir=None):
    input_dir = _normalize_io_dir(input_dir, 'vis/data')
    output_dir = _normalize_io_dir(output_dir, 'vis/pred')
    os.makedirs(output_dir, exist_ok=True)

    if not index_list:
        index_list = _scan_indices(input_dir)
    if not index_list:
        raise FileNotFoundError(f'No complete left/right/sparse/gt sample was found in {input_dir} ')

    config = _load_config(config_name)
    if iter_num is not None:
        config.num_iters = iter_num
    config.workspace = workspace_path
    config.input_dir = input_dir
    if config.dataset == 'vkitti2':
        config.z_max = 102
        config.D = 51
    elif not hasattr(config, 'z_max'):
        config.z_max = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AirDC(config).to(device)
    if getattr(config, 'ckpt_path', ''):
        load_checkpoint_robust(model, config.ckpt_path, 0, debug_mode=False)
    model.eval()

    nickname = config.get('nickname', config.version)
    for start in range(0, len(index_list), max_batch_size):
        batch_indices = index_list[start:start + max_batch_size]
        batch_list = [_prepare_sample(config, index, input_dir) for index in batch_indices]
        batch_data = custom_collate_fn_with_batch(batch_list)
        batch_data = {key: val.to(device) if not isinstance(val, list) else val
                      for key, val in batch_data.items()}

        with torch.no_grad():
            depth_pred_iters, _, _, _, _ = model(batch_data, split='val')
        depth_pred_iters = [reshape_to_BHW(depth) for depth in depth_pred_iters]

        for batch_pos, index in enumerate(batch_indices):
            single_preds = [pred[batch_pos:batch_pos + 1] for pred in depth_pred_iters]
            single_gt = reshape_to_BHW(batch_data['gt'][batch_pos:batch_pos + 1])
            save_path = os.path.join(output_dir, f'{index:03d}_{config.model_name}_{nickname}.png')
            if iter_vis:
                visualize_multi(
                    config,
                    single_preds,
                    target=single_gt,
                    save_path=save_path,
                    rgb_left=batch_data['left_rgb'][batch_pos:batch_pos + 1],
                    rgb_right=batch_data['right_rgb'][batch_pos:batch_pos + 1],
                    sparse=batch_data['d'][batch_pos:batch_pos + 1],
                )
            else:
                visualize_single(
                    config,
                    single_preds[-1],
                    target=single_gt,
                    save_path=save_path,
                    rgb_left=batch_data['left_rgb'][batch_pos:batch_pos + 1],
                    rgb_right=batch_data['right_rgb'][batch_pos:batch_pos + 1],
                    sparse=batch_data['d'][batch_pos:batch_pos + 1],
                    split='val',
                )
            print(f'AirDC inference completed: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Run AirDC inference on exported single samples.')
    parser.add_argument('--config', default='val_all_att', help='Config name or path.')
    parser.add_argument('--indices', type=int, nargs='*', default=[], help='Sample indices. Empty means auto scan.')
    parser.add_argument('--input-dir', default=None, help='Directory containing {idx}left/right/sparse/gt.png files.')
    parser.add_argument('--output-dir', default=None, help='Directory for visualization outputs.')
    parser.add_argument('--iter-num', type=int, default=None, help='Override num_iters.')
    parser.add_argument('--iter-vis', action='store_true', help='Save all iteration predictions.')
    parser.add_argument('--max-batch-size', type=int, default=5)
    args = parser.parse_args()
    inference_single(
        args.indices,
        config_name=args.config,
        iter_num=args.iter_num,
        iter_vis=args.iter_vis,
        max_batch_size=args.max_batch_size,
        output_dir=args.output_dir,
        input_dir=args.input_dir,
    )


if __name__ == '__main__':
    main()
