import torch
import time
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import csv
import os
import sys
import shutil
import inspect
import datetime
from pathlib import Path
torch.backends.cudnn.benchmark = True
import swanlab
import yaml
import argparse

# Define a Configuration class
class Config:
    def __init__(self,** kwargs):
        for key, value in kwargs.items():
            # Recursively convert dictionaries to Config objects.
            if isinstance(value, dict):
                setattr(self, key, Config(**value))
            elif isinstance(value, list):
                # Recursively convert dictionaries inside lists.
                setattr(self, key, [
                    Config(**v) if isinstance(v, dict) else v
                    for v in value
                ])
            else:
                setattr(self, key, value)

    def get(self, key, default=None):
        """Return an attribute value or a default when the key is missing."""
        return getattr(self, key, default)

    # Support the 'key in config' operation.
    def __contains__(self, key):
        return hasattr(self, key)

    # Support config['key'] access.
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'Config' object has no attribute '{key}'")

    # Support keys().
    def keys(self):
        """Return all configuration keys."""
        # Filter special attributes that start with double underscores.
        return [key for key in vars(self).keys() if not key.startswith('__')]

    # Support items().
    def items(self):
        """Return all configuration key-value pairs."""
        return [(key, getattr(self, key)) for key in self.keys()]

    # Support values().
    def values(self):
        """Return all configuration values."""
        return [getattr(self, key) for key in self.keys()]

    # Support iteration over keys.
    def __iter__(self):
        """Iterate over configuration keys."""
        return iter(self.keys())

    # Support dictionary conversion.
    def to_dict(self):
        """Convert the Config object to a nested dictionary."""
        result = {}
        for key in self.keys():
            value = getattr(self, key)
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

def parse_my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--workspace', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--no_debug', action='store_false', dest='debug', default=True)  # Enabled by default; --no_debug disables debug mode.
    parser.add_argument('--train_batchsize', type=int, default=7)
    parser.add_argument('--val_batchsize', type=int, default=10)
    parser.add_argument('--vis_result', action='store_true', help='Enable visualization of validation results')
    parser.add_argument('--vis_ratio', type=float, default=None, help='Percentage of validation samples to visualize (e.g., 0.1 for 10%)')
    parser.add_argument('--train_vis_ratio', type=float, default=None, help='Percentage of training samples to visualize (e.g., 0.001)')
    parser.add_argument('--override', nargs='+', help='Override specific config keys in key=value format', default=[])
    parser.add_argument('--load_from_single', type=str, default=None, help='Load inputs from a single specified folder (e.g. ./demo) instead of using the full dataset')

    parse_config, _ = parser.parse_known_args()
    config_path = parse_config.config_path
    workspace = parse_config.workspace
    log_dir = os.path.join(workspace, parse_config.log_dir)
    config_path=os.path.join(workspace, "config", config_path)

    # Load the YAML file
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    # Automatically map argparse vis_result to config if it was provided as true
    if parse_config.vis_result:
        config_dict['vis_result'] = True
        
    if parse_config.vis_ratio is not None:
        config_dict['vis_ratio'] = parse_config.vis_ratio
        
    if parse_config.train_vis_ratio is not None:
        config_dict['train_vis_ratio'] = parse_config.train_vis_ratio

    # Apply overrides
    for override in parse_config.override:
        if '=' in override:
            key, value = override.split('=', 1)
            # Try to convert value to bool/int/float if possible
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            else:
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
            config_dict[key] = value

    print(f"✅ Config loaded from {config_path}")

    config = Config(**config_dict)
    config.workspace=workspace
    if isinstance(config.ckpt_path, list):
        config.ckpt_path = [ckpt_path if os.path.isabs(ckpt_path) else os.path.join(log_dir, ckpt_path) for ckpt_path in config.ckpt_path]
    else:
        if config.ckpt_path:
            config.ckpt_path = config.ckpt_path if os.path.isabs(config.ckpt_path) else os.path.join(log_dir, config.ckpt_path)
    config.config_path=config_path
    config.train_batchsize = parse_config.train_batchsize
    config.val_batchsize = parse_config.val_batchsize
    config.lr=float(config.lr)
    config.log_dir=os.path.join(log_dir,config.version)
    config.load_from_single = parse_config.load_from_single

    export_to_swanlab = bool(config.get("export_to_swanlab", False))

    os.environ["EXPORT_TO_SWAN"] = "1" if export_to_swanlab and not parse_config.debug else "0"

    return parse_config,config


def swan_lab_init(config):
    """
    Initialize SwanLab and upload fixed hyperparameters for the current run.
    Args:
        config: Parsed configuration object containing training hyperparameters.
    Returns:
        run: Run object returned by swanlab.init for subsequent logging.
    """

    # Build a SwanLab config dictionary from fixed settings.
    swan_config = vars(config)
    # Create a new SwanLab run.
    run = swanlab.init(
        project="DepthCompletion",  # SwanLab project name.
        experiment_name=f"{config.version}_{config.model_name}_{config.dataset}",
        config=swan_config
    )

    return run

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def backup_code(src_folders, dest_folder):
    # Create the backup directory.
    code_backup_folder = os.path.join(dest_folder, "code_backup")
    os.makedirs(code_backup_folder, exist_ok=True)

    # Iterate over each source folder.
    for src_path in src_folders:
        # Ensure the source path is a Path object.
        if src_path.is_file():  # File input.
            if src_path.suffix == '.yaml'or src_path.suffix == '.py':  # Back up Python and YAML files only.
                dest_file = os.path.join(code_backup_folder, src_path.name)
                shutil.copy(src_path, dest_file)
        else:
            src_path = Path(src_path)

            # Use the folder name to create the backup subdirectory.
            folder_name = src_path.name
            dest_subfolder = os.path.join(code_backup_folder, folder_name)
            os.makedirs(dest_subfolder, exist_ok=True)

            # Iterate over Python files in the source folder.
            for file in src_path.rglob('*.py'):
                # Compute the path relative to the source folder.
                relative_path = file.relative_to(src_path)
                # Mirror the directory structure in the destination.
                dest_file = os.path.join(dest_subfolder, relative_path)
                os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                # Copy the file.
                shutil.copy(file, dest_file)

def get_module_path(module):
    return os.path.abspath(inspect.getfile(module))

def print_on_master(ep, epochs, batch_idx, total, start_time, lr):
    elapsed_time = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    if batch_idx > 0:
        estimated_total = elapsed_time / (batch_idx) * total
        eta_str = time.strftime("%H:%M:%S", time.gmtime(estimated_total - elapsed_time))
    else:
        eta_str = "?"

    current_progress = batch_idx / total
    progress_bar_len = 40
    progress_fill = int(current_progress * progress_bar_len)
    progress_bar = "=" * progress_fill + " " * (progress_bar_len - progress_fill)

    print(
        f"\rEpoch: [{ep}/{epochs}] [{batch_idx}/{total}] | {progress_bar} | Elapsed: {elapsed_str} | ETA: {eta_str} | LR: {lr}",
        end="")

def append_to_csv(csv_path, data,header=None):
    """
        Append data to a CSV file and create missing paths when necessary.

        Args:
            csv_path : str  - CSV file path
            data     : list - Data row to write as a list
            header   : list - Optional header used when creating a new file
        """
    # Create the parent directory when it does not exist.
    dir_path = os.path.dirname(csv_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)  # Create parent directories.

    # Check whether the file exists.
    file_exists = os.path.isfile(csv_path)
    # Append data and create the file if necessary.
    with open(csv_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header when creating a new file.
        if not file_exists and header:
            writer.writerow(header)
        writer.writerow(data)

def load_checkpoint(ckpt_path, model, optimizer, scheduler):
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        epoch = ckpt["epoch"]
        batch_idx = ckpt.get("batch_idx", 0)  # Get batch_idx with a default value of 0.
        best_val_loss = ckpt.get("loss", float('inf'))
        print(f"Loaded checkpoint from {ckpt_path}, starting from epoch {epoch}, batch {batch_idx}")
        return epoch, batch_idx, best_val_loss
    return 0, 0, float('inf')

def ensure_dtype(tensor, target_dtype):
    if tensor.dtype != target_dtype:
        tensor = tensor.to(target_dtype)
    return tensor



import socket
def find_free_port(start_port=12355, max_retries=10):
    """Find an available port dynamically."""
    for port in range(start_port, start_port + max_retries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Failed to find an available port in range {start_port}-{start_port+max_retries} ")


best_val_loss = float('inf')
import platform

# Initialize distributed training and return local rank, global rank, and CUDA device.
def init_distributed():
    if platform.system() == 'Windows':
        if torch.cuda.is_available() or torch.cuda.device_count() == 1:
            return 0,0, torch.device("cuda:0")  # Return directly for single-GPU execution.

    # Check if we are actually running under DDP (torchrun / launch)
    if 'RANK' not in os.environ or 'LOCAL_RANK' not in os.environ:
        print("Not running under DDP (no RANK/LOCAL_RANK found), falling back to single GPU mode.")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            return 0, 0, torch.device("cuda:0")
        else:
            return 0, 0, torch.device("cpu")

    # Use Gloo on Windows.
    # Gloo is cross-platform; NCCL is optimized for NVIDIA GPUs on Linux.
    backend = "gloo" if platform.system() == 'Windows' else "nccl"

    try:
        # Read distributed environment variables.
        init_method = 'env://'
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
        )
        # Read the local GPU index from environment variables.
        local_rank = int(os.environ['LOCAL_RANK'])
        # Read the global process rank.
        rank = dist.get_rank()
        # Bind each process to its assigned GPU.
        torch.cuda.set_device(local_rank)
        return local_rank, rank, torch.device("cuda", local_rank)

    except RuntimeError as e:
        if "address already in use" in str(e).lower():
            print("Port conflict detected; trying another port...")
            os.environ['MASTER_PORT'] = str(int(os.environ['MASTER_PORT']) + 1)
            return init_distributed()  # Retry recursively.
        if "10049" in str(e):
            # Repair the Winsock protocol stack.
            subprocess.run(['netsh', 'winsock', 'reset'], check=True, shell=True)
            subprocess.run(['netsh', 'int', 'ip', 'reset'], check=True, shell=True)
            os.environ['MASTER_PORT'] = str(int(os.environ['MASTER_PORT']) + 1)
            return init_distributed()  # Retry recursively.
        raise

def setup_model(model, local_rank,find_unused_parameters=True):
    # Move the model to the selected device.
    model = model.to(local_rank)

    # Check whether distributed training is active.
    if dist.is_initialized() and dist.get_world_size() > 1:
        # Wrap the model with DDP.
        model = DDP(model,
                    device_ids=[local_rank], # GPU assigned to the current process.
                    output_device=local_rank, # Place outputs on the assigned GPU.
                    find_unused_parameters=find_unused_parameters # Detect unused parameters to avoid DDP errors.
                    )
    return model

def custom_collate_fn_with_batch(batch):
    # Extract points_d and d_index and compute the maximum point count.
    batch_points_d = [
        torch.tensor(item["points_d"].copy()) if not isinstance(item["points_d"], torch.Tensor) else item["points_d"]
        for item in batch]
    batch_d_index = [
        torch.tensor(item["d_index"].copy()) if not isinstance(item["d_index"], torch.Tensor) else item["d_index"] for
        item in batch]
    batch_points_gt = [
        torch.tensor(item["points_gt"].copy()) if not isinstance(item["points_gt"], torch.Tensor) else item["points_gt"]
        for item in batch]
    batch_gt_index = [
        torch.tensor(item["gt_index"].copy()) if not isinstance(item["gt_index"], torch.Tensor) else item["gt_index"]
        for item in batch]


    max_points_size_d = max([pts.shape[0] for pts in batch_points_d])
    max_points_size_gt = max([pts.shape[0] for pts in batch_points_gt])
    # Initialize padded tensors for points_d, mask, and d_index.
    batch_points_d_padded = torch.zeros((len(batch), max_points_size_d, 3), dtype=torch.float16)
    batch_points_gt_padded = torch.zeros((len(batch), max_points_size_gt, 3), dtype=torch.float16)
    mask_d = torch.zeros((len(batch), max_points_size_d, 1), dtype=torch.bool)
    mask_gt = torch.zeros((len(batch), max_points_size_gt, 1), dtype=torch.bool)
    batch_d_index_padded = torch.full((len(batch), max_points_size_d, 2), 10000, dtype=torch.long)
    batch_gt_index_padded = torch.full((len(batch), max_points_size_gt, 2), 0, dtype=torch.long)

    # Pad points_d and d_index.
    for i, (points_d, d_index) in enumerate(zip(batch_points_d, batch_d_index)):
        num_points = points_d.shape[0]
        batch_points_d_padded[i, :num_points, :] = points_d
        mask_d[i, :num_points, :] = 1
        batch_d_index_padded[i, :num_points, :] = d_index
    for i, (points_gt, gt_index) in enumerate(zip(batch_points_gt, batch_gt_index)):
        num_points = points_gt.shape[0]
        batch_points_gt_padded[i, :num_points, :] = points_gt
        mask_gt[i, :num_points, :] = 1
        batch_gt_index_padded[i, :num_points, :] = gt_index

    # Collate fields other than points_d and d_index.
    collated_batch = {
        "points_d": batch_points_d_padded,
        "mask_d": mask_d,
        "d_index": batch_d_index_padded,
        "gt_index": batch_gt_index_padded,
        "points_gt": batch_points_gt_padded,
        "mask_gt": mask_gt
    }
    for key in batch[0].keys():
        if key not in ["points_d", "d_index","points_gt", "gt_index"]:
            if key in ["date"]:
                collated_batch[key] = [item[key] for item in batch]  # Keep the field unchanged without tensor conversion.
            else:
            # Stack tensors directly or convert values to tensors before stacking.
                collated_batch[key] = torch.stack([
                    # Clone and detach tensor values.
                    item[key].clone().detach() if isinstance(item[key], torch.Tensor) else
                    # Convert NumPy arrays or lists to float16 tensors.
                    torch.tensor(item[key], dtype=torch.float16) if isinstance(item[key], (np.ndarray, list)) else
                    # Convert other scalar values to tensors.
                    torch.tensor(item[key])
                    for item in batch
                ])
    return collated_batch

def check_feature_maps(feature_maps, name):
    for i, fm in enumerate(feature_maps):
        if torch.isnan(fm).any():
            print(f"NaN detected in {name} level {i}")
            print(f"Feature map stats: min={fm.min()}, max={fm.max()}, mean={fm.mean()}, std={fm.std()}")
        if torch.isinf(fm).any():
            print(f"Inf detected in {name} level {i}")
            print(f"Feature map stats: min={fm.min()}, max={fm.max()}, mean={fm.mean()}, std={fm.std()}")

def save_checkpoint(model, optimizer, scheduler, scaler, ep, batch_idx, train_loss, save_path):
    # Ensure model parameters are in fp32.
    model_state = {k: v.to(torch.float16) for k, v in model.state_dict().items()}
    checkpoint = {
        'epoch': ep,
        'batch_idx': batch_idx,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'train_loss': train_loss / (batch_idx + 1)
    }
    torch.save(checkpoint, save_path)

def model_load_checkpoint(model, optimizer, scheduler, scaler, load_path, device,strict=True):
    # Load the checkpoint file.
    checkpoint = torch.load(load_path, map_location='cpu')

    # Get state dictionaries from the model and checkpoint.
    model_state_dict = model.state_dict()
    checkpoint_state_dict = checkpoint['model_state_dict']

    # Detect module. prefix alignment between checkpoint and model keys.
    def adjust_keys(state_dict, add_module=False, remove_module=False):
        if add_module:
            return {f'module.{k}': v for k, v in state_dict.items()}
        elif remove_module:
            return {k[7:]: v for k, v in state_dict.items()}
        else:
            return state_dict

    # Check whether the model state_dict uses the module. prefix.
    has_module_prefix = any(k.startswith('module.') for k in model_state_dict.keys())

    # Check whether the checkpoint state_dict uses the module. prefix.
    has_checkpoint_module_prefix = any(k.startswith('module.') for k in checkpoint_state_dict.keys())

    # Adjust checkpoint keys according to prefix compatibility.
    if has_module_prefix and not has_checkpoint_module_prefix:
        # Add the module. prefix when the model expects it.
        checkpoint_state_dict = adjust_keys(checkpoint_state_dict, add_module=True)
    elif not has_module_prefix and has_checkpoint_module_prefix:
        # Remove the module. prefix when the model does not expect it.
        checkpoint_state_dict = adjust_keys(checkpoint_state_dict, remove_module=True)

    # Load model parameters.
    model.load_state_dict(checkpoint_state_dict, strict=strict)

    # Move the model to the selected device after loading.
    model = model.to(device)
    if hasattr(model, 'convert_to_fp16'):
        model.convert_to_fp16()

    # Load optimizer, scheduler, and scaler states.
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Count the number of parameters not loaded
    missing_keys = []
    unexpected_keys = []

    # Compare checkpoint and model state dicts
    for k in checkpoint_state_dict:
        if k not in model_state_dict:
            missing_keys.append(k)
        elif checkpoint_state_dict[k].shape != model_state_dict[k].shape:
            unexpected_keys.append(k)

    # Print a message if all model parameters are loaded
    if len(missing_keys) == 0 and len(unexpected_keys) == 0:
        print("All model params are loaded.")
    else:
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys}")

    return int(checkpoint['epoch']), int(checkpoint['batch_idx'])

def check_shape(rank,name, own_state, param,failed_params):
    try:
        # Check shape compatibility explicitly.
        if own_state[name].shape != param.shape:
            raise RuntimeError(f"Shape mismatch: expected {own_state[name].shape}, got {param.shape}")
        own_state[name].copy_(param)
    except RuntimeError as e:
        if rank == 0:
            err_msg = str(e)
            shape_info = ""
            if "shape" in err_msg.lower():
                expected_shape = own_state[name].shape
                actual_shape = param.shape
                shape_info = f"\n    Expected shape: {list(expected_shape)}  Checkpoint shape: {list(actual_shape)}"
            print(f"× Failed to load {name}: {err_msg}{shape_info}")
        failed_params.append(name)

def load_checkpoint_robust(model,
                           checkpoint_path,
                           rank,
                           optimizer=None,
                           print_not_load_info=False,
                           load_dict=None,
                           not_load_dict=[],
                           debug_mode=False,
                           net_key="model_state_dict",
                           sub="module.",
                           strict=True):
    """
    Load a checkpoint by matching raw_name, sub+raw_name, and names with the sub prefix removed.
    Report unmatched checkpoint keys and model keys that remain unloaded.
    Example sub prefix: "module."
    """

    if checkpoint_path == "":
        if rank == 0:
            choice = input("\nNo available checkpoint!! Continue loading model? (y/n): ").strip().lower()
            while choice not in ['y', 'n']:
                choice = input("Please enter 'y' to continue or 'n' to abort: ").strip().lower()
            should_exit = choice != 'y'
        if dist.is_initialized():
            exit_tensor = torch.tensor([should_exit], dtype=torch.int).cuda()
            dist.broadcast(exit_tensor, src=0)
            should_exit = exit_tensor.item() == 1
        if should_exit:
            if rank == 0:
                print("Aborting model loading...")
            sys.exit(1)

    # Attempt loading when checkpoint_path is provided.
    if checkpoint_path:
        if rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        own_state = model.state_dict()
        failed_ckpt = []        # checkpoint params not matched
        loaded_model_keys = set()  # own_state keys that get loaded

        for raw_name, param in checkpoint.get(net_key, {}).items():
            matched = False
            tried_names = [raw_name]
            if not raw_name.startswith(sub):
                tried_names.append(sub + raw_name)
            else:
                tried_names.append(raw_name[len(sub):])

            for try_name in tried_names:
                if try_name not in own_state or any(bad in try_name for bad in not_load_dict):
                    if print_not_load_info and rank == 0 and try_name in own_state and any(bad in try_name for bad in not_load_dict):
                        print(f"× Skip loading {try_name} because it matches not_load_dict")
                    continue
                if load_dict is not None and not any(good in try_name for good in load_dict):
                    if print_not_load_info and rank == 0:
                        print(f"× {try_name} not in load_dict, skipping")
                    continue
                # Check shapes and copy tensors after a successful key match.
                check_shape(rank,try_name, own_state, param, failed_ckpt)
                loaded_model_keys.add(try_name)
                matched = True
                break

            if not matched:
                failed_ckpt.append(raw_name)
                if print_not_load_info and rank == 0:
                    print(f"× Failed to match any variant of '{raw_name}' in own_state")

        # Print unmatched checkpoint parameters.
        if rank == 0 and failed_ckpt:
            print("\nThe following checkpoint parameters were not loaded due to name or shape mismatch:")
            for p in failed_ckpt:
                print(f"  - {p}")

        # Compute model keys that were not loaded.
        missing_model_keys = [k for k in own_state.keys()
                              if k not in loaded_model_keys
                              and not any(bad in k for bad in not_load_dict)
                              and (load_dict is None or any(good in k for good in load_dict))]
        if rank == 0 and missing_model_keys:
            print("\nThe following model parameters were not loaded because they are missing from the checkpoint:")
            for k in missing_model_keys:
                print(f"  - {k}")

        # Load the updated own_state into the model.
        model.load_state_dict(own_state, strict=strict)
        if rank == 0:
            print(f"✅ Weights loaded (strict={strict})")

        return missing_model_keys


def load_my_checkpoint_first(config, model,rank,debug=False):
    ckpt_paths = config.ckpt_path if isinstance(config.ckpt_path, list) else [config.ckpt_path]

    # Find the first existing checkpoint in order.
    selected_ckpt = None
    for p in ckpt_paths:
        if p and os.path.exists(p):
            selected_ckpt = p
            if rank == 0:
                print(f"✅ Found checkpoint at {p}, using this one.")
            break
        else:
            if rank == 0:
                print(f"⚠️  Checkpoint not found at {p}, trying next...")
    if selected_ckpt is None:
        # Raise an error when no checkpoint exists.
        # raise FileNotFoundError(f"No checkpoint file found in any of: {ckpt_paths}")
        print (f"❌ No checkpoint file found in any of: {ckpt_paths}")
        return
    if config.model_name != "AirDC":
        raise ValueError("model_name must be AirDC")
    if selected_ckpt != "":
        load_checkpoint_robust(model, selected_ckpt, rank, debug_mode=debug)

def normalize_key(key, sub="module."):
    """
    Ensure key has the given sub prefix,
    or return unchanged if already prefixed.
    """
    return key if key.startswith(sub) else sub + key


def merge_checkpoints(ckpt_paths, net_key="model_state_dict", sub="module."):
    """
    For each checkpoint, normalize all state keys first (add sub prefix if missing),
    then merge: later checkpoints override earlier. Return:
      merged: dict[norm_key] = tensor
      source: dict[norm_key] = checkpoint path
      overrides: dict[(new_ckpt, old_ckpt)] = list of norm_keys overridden
    """
    merged = {}
    source = {}
    overrides = {}

    for ckpt in ckpt_paths:
        if not ckpt or not os.path.exists(ckpt):
            continue
        data = torch.load(ckpt, map_location='cpu')
        raw_state = data.get(net_key, {})
        # Normalize all keys up front
        norm_state = {normalize_key(k, sub): v for k, v in raw_state.items()}
        for nk, param in norm_state.items():
            if nk in merged:
                prev = source[nk]
                overrides.setdefault((ckpt, prev), []).append(nk)
            merged[nk] = param
            source[nk] = ckpt

    return merged, source, overrides



def collapse_chain(name, node):
    parts = [name]
    subnode = node
    while len(subnode) == 1:
        child_name, child_node = next(iter(subnode.items()))
        parts.append(child_name)
        subnode = child_node
    return parts, subnode


def build_tree(keys, sub_prefix):
    tree = {}
    for k in keys:
        parts = k.split('.')
        root = '.'.join(parts[:2])
        node = tree.setdefault(root, {})
        for seg in parts[2:]:
            node = node.setdefault(seg, {})
    return tree

def print_sub(node, indent):
    for name, child in sorted(node.items()):
        parts, residual = collapse_chain(name, child)
        full = '.'.join(parts)
        # no children
        if not residual:
            print(' ' * indent + f"- {full}")
        # all children are leaves
        elif all(not v for v in residual.values()):
            leaves = ', '.join(sorted(residual.keys()))
            print(' ' * indent + f"- {full}.({leaves})")
        else:
            # mixed: print node and recurse
            print(' ' * indent + f"- {full}.")
            print_sub(residual, indent + 4)


def print_hierarchy(title, keys, sub="module."):
    if not keys: return
    print(f"\n{title}")
    tree = build_tree(keys, sub)
    for root, subtree in tree.items():
        print(f"{root}.")
        print_sub(subtree, 4)


def load_checkpoints_sequence(model, ckpt_paths, rank, net_key="model_state_dict", sub="module."):
    if not isinstance(ckpt_paths, (list, tuple)):
        ckpt_paths = [ckpt_paths]
    merged, source, overrides = merge_checkpoints(ckpt_paths, net_key, sub)

    # overrides
    if rank == 0 and overrides:
        for (new, old), keys in overrides.items():
            print(f"Override: {new} -> {old}:")
            print_hierarchy("", keys, sub)

    # load into model
    own = model.state_dict()
    norm_model = {normalize_key(k, sub): k for k in own.keys()}
    for nk, param in merged.items():
        orig = norm_model.get(nk)
        if orig and own[orig].shape == param.shape:
            own[orig].copy_(param)
    model.load_state_dict(own, strict=True)

    # model-only and checkpoint-only
    model_only = [orig for nk, orig in norm_model.items() if nk not in merged]
    ckpt_only = [nk for nk in merged if nk not in norm_model]

    if rank == 0:
        print_hierarchy("❌ all checkpoint missed(in model):", model_only, sub)
        print_hierarchy("▶️only checkpoint have(not in model):", ckpt_only, sub)


def load_my_checkpoint_merge(config, model, rank, debug=False):
    ckpts = config.ckpt_path if isinstance(config.ckpt_path, list) else [config.ckpt_path]
    if not any(ckpts):
        raise FileNotFoundError(f"No checkpoint paths provided: {ckpts}")
    load_checkpoints_sequence(
        model, ckpts, rank,
        net_key="model_state_dict",
        sub="module."
    )



def check_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")
        return True
    return False



import random
def set_seed(seed, rank=0):
    """Set all random seeds and deterministic options."""
    seed += rank  # Use different seeds for different ranks.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU execution.
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    """Initialize a DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)
from collections import defaultdict
class LayerTimer:
    def __init__(self, model):
        self.model = model
        self.timings = defaultdict(list)
        self.handles = []
        self.start_times = {}  # Store per-module start times.
        self._register_hooks()

    def _record_time(self, module, inputs, outputs):
        # Synchronize CUDA operations.
        start_time = self.start_times.pop(module, None)
        if start_time is not None:
            elapsed = time.perf_counter() - start_time
            module_name = self._get_module_name(module)
            self.timings[module_name].append(elapsed)

    def _get_module_name(self, module):
        for name, m in self.model.named_modules():
            if m is module:
                return name
        return str(module.__class__.__name__)

    def _register_hooks(self):
        def pre_hook(module, inputs):
            # Synchronize CUDA and record the start time.
            self.start_times[module] = time.perf_counter()

        # Register hooks for all child modules.
        for name, module in self.model.named_modules():
            handle_pre = module.register_forward_pre_hook(pre_hook)
            handle_post = module.register_forward_hook(self._record_time)
            self.handles.extend([handle_pre, handle_post])

    def get_timings(self):
        return {k: sum(v) / len(v) * 1000 for k, v in self.timings.items()}  # Average runtime in milliseconds.
    def reset_timings(self):
        """Clear timing data while keeping hooks."""
        self.timings.clear()
        self.start_times.clear()

    def clear(self):
        """Remove hooks and clear timing data."""
        self.reset_timings()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
