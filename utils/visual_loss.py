from utils import *
import torch.nn.functional as F
import os
#####=========Visualization utilities.=========#####
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch.nn as nn
import swanlab

def normalize_for_display(rgb):
    # Map image values from [-2, 2] to [0, 1].
    min_val = np.min(rgb)
    max_val = np.max(rgb)
    rgb_normalized = (rgb - min_val) / (max_val - min_val) * 255  # Map to [0, 255].
    return rgb_normalized

def visualize_multi(
    args,
    pred_depth_iters,           # Accepts a list of tensors or a single tensor.
    target,
    save_path='tight_compare.png',
    rgb_left=None,
    rgb_right=None,
    sparse=None,
):
    """
    Vertical multi-row, two-column visualization.
      Row 0: RGB Left | RGB Right
      Row 1: Sparse   | Ground Truth
      Row 2+i: Pred_i (colored) | Error Map_i (with RMSE shown in the title)
    Display only the first sample in the batch.
    """
    # Ensure pred_depth_iters is a list.
    if not isinstance(pred_depth_iters, (list, tuple)):
        pred_depth_iters = [pred_depth_iters]

    n_iters = len(pred_depth_iters)
    rows = 2 + n_iters
    idx = 0  # Use the first sample only.
    height_ratios = [1] * rows  # Use equal row heights.
    # Create the subplot canvas.
    fig, axs = plt.subplots(
        rows, 2,
        figsize=(10, 1.2 * rows),
        gridspec_kw={
            'wspace': 0.02,
            'hspace': 0,
            'height_ratios': height_ratios
        },
        constrained_layout=True
    )

    for ax in axs.flatten():
        ax.axis('off')
    # Row 0: left: RGB Left, right: RGB Right
    if rgb_left is not None:
        img = normalize_for_display(rgb_left[idx].cpu().numpy().transpose(1,2,0)).astype(np.uint8)
        axs[0,0].imshow(img)
        axs[0,0].set_title('RGB Left', fontsize=10, pad=2)
    axs[0,0].axis('off')

    if rgb_right is not None:
        img = normalize_for_display(rgb_right[idx].cpu().numpy().transpose(1,2,0)).astype(np.uint8)
        axs[0,1].imshow(img)
        axs[0,1].set_title('RGB Right', fontsize=10, pad=2)
    axs[0,1].axis('off')

    # Row 1: left: Sparse Depth, right: Ground Truth
    if sparse is not None:
        sp = sparse[idx].squeeze()
        sp_col = apply_colormap_to_depth(sp, stretch_range=(0, args.z_max)).squeeze(0)
        axs[1,0].imshow(sp_col)
        axs[1,0].set_title(f'Sparse Depth ({args.lidar_lines} lines)', fontsize=10, pad=2)
    axs[1,0].axis('off')

    gt = target[idx].squeeze()
    gt_col = apply_colormap_to_depth(gt, stretch_range=(0, args.z_max)).squeeze(0)
    axs[1,1].imshow(gt_col)
    axs[1,1].set_title('Ground Truth', fontsize=10, pad=2)
    axs[1,1].axis('off')

    # Rows 2...: plot Prediction and Error Map for each prediction iteration.
    for i, pred in enumerate(pred_depth_iters):
        row = 2 + i
        pred_map = pred[idx].squeeze()
        pred_col = apply_colormap_to_depth(pred_map, stretch_range=(0, args.z_max)).squeeze(0)
        # Left: predicted depth map.
        axs[row,0].imshow(pred_col)
        axs[row,0].set_title(f'Iteration {i+1}', fontsize=10, pad=2)
        axs[row,0].axis('off')

        # Right: error map with RMSE.
        error_map, error_cmap, error_norm = apply_log_error_map(pred_map, gt)
        rmse = RMSE(pred_map, gt)
        im = axs[row,1].imshow(error_map, cmap=error_cmap, norm=error_norm)
        axs[row,1].set_title(f'Error {i+1} (RMSE: {rmse:.2f} mm)', fontsize=10, pad=2)
        axs[row,1].axis('off')

        bounds = [0, 1, 2, 3, 4, 5, 10]
        cbar = fig.colorbar(im,
                            ax=axs[row, 1],
                            boundaries=bounds,
                            ticks=bounds,
                            fraction=0.066,  # width of cbar as fraction of Axes
                            pad=0.04)  # space between image and cbar
        cbar.set_label('Relative Error (%)', rotation=270, labelpad=12, fontsize=8)
        cbar.ax.tick_params(labelsize=8)


    # fig.suptitle(f'Visualization for Sample {idx + 1}', fontsize=14, y=0.98)

    # Ensure the save directory exists.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(fig)

def visualize_single(args, pred_depth, target, save_path='tight_compare.png',
                     rgb_left=None, rgb_right=None, sparse=None, split='train'):
    """
    Vertical two-column visualization with RGB images and depth-related maps.
    Display only the first sample in the batch.
    """
    # Use the first sample in the batch.
    idx = 0

    # Create a vertical two-column subplot layout.
    fig, axs = plt.subplots(
        3, 2,  # four-row, two-column layout
        figsize=(10,4) if args.oheight < 300 else (10,5),
        gridspec_kw={'wspace': 0.02, 'hspace': 0.02} , # Remove gaps between rows and columns.
        constrained_layout = True
    )

    # First column: RGB images.------------------------------------------

    # Top-left: left RGB image.
    if rgb_left is not None:
        rgb_img = normalize_for_display(
            rgb_left[idx].cpu().numpy().transpose(1, 2, 0)
        ).astype(np.uint8)
        axs[0, 0].imshow(rgb_img)
        axs[0, 0].set_title('RGB Left', fontsize=10, pad=5)
    axs[0, 0].axis('off')

    # Bottom-left: right RGB image.
    if rgb_right is not None:
        rgb_img = normalize_for_display(
            rgb_right[idx].cpu().numpy().transpose(1, 2, 0)
        ).astype(np.uint8)
        axs[0, 1].imshow(rgb_img)
        axs[0, 1].set_title('RGB Right', fontsize=10, pad=5)
    axs[0, 1].axis('off')

    # Second column: depth-related maps.----------------------------------------

    # First row: sparse depth map.
    if sparse is not None:
        sparse_depth = sparse[idx].squeeze()
        sparse_colored = apply_colormap_to_depth(sparse_depth, stretch_range=(0, args.z_max)).squeeze(0)
        axs[1, 0].imshow(sparse_colored)
        axs[1, 0].set_title(f'Sparse Depth ({args.lidar_lines} Lines)', fontsize=10, pad=5)
    axs[1, 0].axis('off')

    # Second row: predicted depth map.
    pred = pred_depth[idx].squeeze()
    pred_colored = apply_colormap_to_depth(pred, stretch_range=(0, args.z_max)).squeeze(0)
    axs[1, 1].imshow(pred_colored)
    axs[1, 1].set_title('Prediction', fontsize=10, pad=5)
    axs[1, 1].axis('off')

    # Third row: ground-truth depth map.
    gt = target[idx].squeeze()
    gt_colored = apply_colormap_to_depth(gt, stretch_range=(0, args.z_max)).squeeze(0)
    axs[2, 0].imshow(gt_colored)
    axs[2, 0].set_title('Ground Truth', fontsize=10, pad=5)
    axs[2, 0].axis('off')

    # Fourth row: error map.
    error_map, error_cmap, error_norm = apply_log_error_map(pred, gt)
    rmse = RMSE(pred, gt)
    im = axs[2, 1].imshow(error_map, cmap=error_cmap, norm=error_norm)
    axs[2, 1].set_title(f'Error Map (RMSE: {rmse:.2f} mm)', fontsize=10, pad=5)
    axs[2, 1].axis('off')

    # Add the colorbar only for the error map.
    bounds = [0, 1, 2, 3, 5, 7.5, 10, 15, 20]
    cbar = fig.colorbar(im,
                        ax=axs[2, 1],
                        boundaries=bounds,
                        ticks=bounds,
                        fraction=0.066,  # width of cbar as fraction of Axes
                        pad=0.04)  # space between image and cbar
    cbar.set_label('Relative Error (%)', rotation=270, labelpad=12, fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    # Set the optional figure title.
    # fig.suptitle(f'Visualization for Sample {idx + 1}', fontsize=14, y=0.98)


    # Ensure the save path exists.
    dir_path = os.path.dirname(save_path)
    if dir_path != "":
        os.makedirs(dir_path, exist_ok=True)
    # plt.tight_layout()
    # Save the figure.
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(fig)


def visualize_all(args,pred_depth, target, save_path='tight_compare.png',
                  rgb_left=None,rgb_right=None, sparse=None,split='train'):
    """
    Compact visualization with RGB, sparse depth, prediction, ground truth, and error map columns.
    Display all samples in a batch from top to bottom.
    """
    batch_size = target.shape[0]

    # Create the subplot canvas with a gap-free grid layout.
    if split == 'train':
        fig, axs = plt.subplots(
            batch_size, 6,
            figsize=(10, 1.3 * batch_size) if args.oheight<300 else (10,1.6*batch_size),  # Use a smaller height factor for compact rows.
            gridspec_kw={'wspace': 0, 'hspace': 0}  # Remove gaps between rows and columns.
        )
    else:
        if args.oheight > 300:
            fig_width=23
            fig_height=1.6 * batch_size
        else:
            fig_width=26
            fig_height=1.4 * batch_size
        fig, axs = plt.subplots(
            batch_size, 6,
            figsize=(fig_width, fig_height),  # Use a smaller height factor for compact rows.
            gridspec_kw={'wspace': 0, 'hspace': 0}  # Remove gaps between rows and columns.
        )

    # Adjust axes dimensions when batch_size is 1.
    if batch_size == 1:
        axs = np.expand_dims(axs, 0)
    if args.oheight > 300:
        texty=0.95
    else:
        texty=0.92
    for i in range(batch_size):
        # First column: left RGB image. ----------------------------------------------------------
        if rgb_left is not None:
            rgb_img = normalize_for_display(
                rgb_left[i].cpu().numpy().transpose(1, 2, 0)
            ).astype(np.uint8)
            axs[i, 0].imshow(rgb_img)

        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'RGB_left_Sample {i + 1}', fontsize=5, y=texty,color='black')  # Use a smaller title font and move it upward.

        if rgb_right is not None:
            rgb_img = normalize_for_display(
                rgb_right[i].cpu().numpy().transpose(1, 2, 0)
            ).astype(np.uint8)
            axs[i, 1].imshow(rgb_img)

        axs[i, 1].axis('off')
        axs[i, 1].set_title(f'RGB_Right_Sample {i + 1}', fontsize=5, y=texty,color='black')  # Use a smaller title font and move it upward.

        # Second column: sparse depth.
        if sparse is not None:
            sparse_depth = sparse[i].squeeze()
            sparse_colored = apply_colormap_to_depth(sparse_depth, stretch_range=(0, args.z_max)).squeeze(0)
            axs[i, 2].imshow(sparse_colored)
        axs[i, 2].axis('off')
        axs[i, 2].set_title(f'Sparse Depth({args.lidar_lines} Lines)', fontsize=5, y=texty,color='black')

        # Third column: predicted depth. ---------------------------------------------------------
        pred = pred_depth[i].squeeze()
        pred_colored = apply_colormap_to_depth(pred, stretch_range=(0, args.z_max)).squeeze(0)
        axs[i, 3].imshow(pred_colored)
        axs[i, 3].axis('off')
        axs[i, 3].set_title('Prediction', fontsize=5, y=texty,color='black')

        # Fourth column: ground-truth depth. ---------------------------------------------------------
        gt = target[i].squeeze()
        gt_colored = apply_colormap_to_depth(gt, stretch_range=(0, args.z_max)).squeeze(0)
        axs[i, 4].imshow(gt_colored)
        axs[i, 4].axis('off')
        axs[i, 4].set_title('Ground Truth', fontsize=5, y=texty,color='black')

        # Fifth column: error map. -------------------------------------------------------
        error_map, error_cmap, error_norm = apply_log_error_map(pred, gt)
        rmse = RMSE(pred, gt)
        im = axs[i, 5].imshow(error_map, cmap=error_cmap, norm=error_norm)
        axs[i, 5].axis('off')
        axs[i, 5].set_title(f'Error Map(RMSE: {rmse:.2f} mm)', fontsize=5, y=texty,color='black')

    # Configure the layout to fill the canvas.
    plt.subplots_adjust(
        left=0.01, right=0.99,
        top=0.95 - 0.02 * batch_size,  # Adjust top spacing dynamically.
        bottom=0.01,
        wspace=0, hspace=0
    )

    # Use a tight layout when saving.
    dir_path = os.path.dirname(save_path)
    if dir_path != "":  # Avoid empty directory paths such as save_path="image.png".
        os.makedirs(dir_path, exist_ok=True)  # Create directories in the save path.
    plt.savefig(save_path, dpi=450, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

def apply_log_error_map(coarse, target):
    """
    Generate an error map with a custom colormap and invalid-region handling.

    Args:
        coarse: (H, W) Coarse prediction
        target: (H, W) Ground truth

    Returns:
        error_map: Final error-map array (H, W)
        cmap: custom colormap
        norm: Normalization object for color mapping
    """
    # Ensure inputs are PyTorch tensors.
    coarse = torch.as_tensor(coarse)
    target = torch.as_tensor(target)

    # Create a valid-pixel mask; target <= 0 is invalid.
    valid_mask = (target > 0).float()

    # Compute relative error in percent and mask invalid regions.
    relative_error = torch.abs(coarse - target) / target
    relative_error[valid_mask == 0] = -1  # Set invalid regions to -1.
    relative_error=relative_error * 100

    # Convert to NumPy format.
    error_map = relative_error.detach().cpu().numpy()
    from scipy.ndimage import binary_dilation, grey_dilation

    # Dilate non-background pixels, i.e., pixels not equal to -1.
    non_background_mask = (error_map != -1)  # Extract non-background regions.

    # Dilate non-background regions with binary_dilation.
    dilated_mask = binary_dilation(non_background_mask, structure=np.ones((3, 3)))  # 3x3 dilation operation.

    # Use grey_dilation to fill dilated regions with local maxima.
    dilated_values = grey_dilation(error_map, size=(3, 3))

    # Initialize a new error map with background kept as -1.
    expanded_error_map = np.full_like(error_map, -1)
    # Assign grey_dilation values to the dilated non-background region.
    expanded_error_map[dilated_mask] = dilated_values[dilated_mask]

    # Custom colormap with 8 gradient bins.
    colors = [
        (0, 0, 0),       # black for invalid background (-1)
        (30, 30, 120),   # dark blue (1% error)
        (49, 54, 155),    # blue-violet (3% error)
        (77, 126, 185), # light blue (5% error)
        (171, 217, 233),   # sky blue (10% error)
        (254, 224, 144),   # yellow (15% error)
        (255, 140, 0),   # orange-yellow (20% error)
        (253, 174, 97),    # orange-red (25% error)
        (215, 50, 40),   # red (30% or highererror)
    ]
    # Normalize RGB colors to floating-point values in [0, 1].
    colors = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in colors]
    # Create a ListedColormap from custom colors.
    cmap = mcolors.ListedColormap(colors)

    # Create a BoundaryNorm for each color interval.
    bounds = [-1, 0, 1, 2, 3, 5, 7.5, 10, 15, 20]  # -1 maps to black; remaining values use 8 gradient bins.
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Return the processed error map, colormap, and normalization object.
    return expanded_error_map, cmap, norm

def apply_colormap_to_depth(depth, stretch_range=None, colormap=plt.cm.jet):
    """
    Apply a colormap to a depth map and normalize depth values to a fixed range.

    Parameters:
    - depth: (B, H, W) tensor containing depth values.
    - stretch_range: tuple of (min_depth, max_depth) to normalize the depth.
    - colormap: Matplotlib colormap to use for coloring.

    Returns:
    - colored_depth_rgb: (B, H, W, 3) numpy array with RGB color mapped depth.
    """
    # depth(B, H, W)
    if depth.dim()==4:
        depth = depth.squeeze(1)  # Remove channel dimension
    if depth.dim()== 2:
        depth = depth.unsqueeze(0)  # Add batch dimension
    assert depth.dim() == 3, "Depth should have 3 dimensions (B, H, W)"
    # Move tensor to CPU and convert to numpy array
    depth_np = depth.cpu().detach().numpy()

    if stretch_range is None:
        min_depth = depth_np.min()
        max_depth = depth_np.max()
    else:
        min_depth, max_depth = stretch_range

    # Handle the case where min_depth equals max_depth
    if min_depth == max_depth:
        normalized_depth = np.zeros_like(depth_np)  # Assign a default value, here 0
    else:
        # Normalize depth to [0, 1]
        normalized_depth = (depth_np - min_depth) / (max_depth - min_depth)
        normalized_depth = np.clip(normalized_depth, 0, 1)  # Clip to ensure values are within [0, 1]

    normalized_depth[depth_np == -1] = 0  # Set background to black

    # Apply colormap
    colored_depth = colormap(normalized_depth)

    # Convert to RGB by discarding the alpha channel
    colored_depth_rgb = (colored_depth[:, :, :, :3] * 255).astype(np.uint8)

    return colored_depth_rgb


def apply_colormap_to_depth_numpy(depth_nd, stretch_range=None, colormap=plt.cm.jet):
    """
    Colorize depth maps with NumPy.

    Parameters:
    - depth_nd: numpy ndarray with shape (H, W) or (B, H, W), containing depth values.
    - stretch_range: (min_depth, max_depth) tuple for depth normalization; if None, infer min and max values.
    - colormap: Matplotlib colormap used for color mapping.

    Returns:
    - colored_depth_rgb: uint8 RGB colorized depth map with shape (B, H, W, 3).
    """
    # Ensure the input is an ndarray.
    if not isinstance(depth_nd, np.ndarray):
        raise TypeError("depth_nd must be a numpy ndarray")

    # Convert to 3D: (B, H, W).
    if depth_nd.ndim == 2:
        depth = depth_nd[np.newaxis, ...]  # (1, H, W)
    elif depth_nd.ndim == 3:
        depth = depth_nd
    else:
        raise ValueError("depth_nd must have shape (H, W) or (B, H, W)")

    # Compute the normalization range.
    if stretch_range is None:
        min_depth = np.min(depth[np.where(depth != -1)]) if np.any(depth != -1) else 0.0
        max_depth = np.max(depth[np.where(depth != -1)]) if np.any(depth != -1) else 1.0
    else:
        min_depth, max_depth = stretch_range

    # Handle the min_depth == max_depth case.
    if min_depth == max_depth:
        normalized = np.zeros_like(depth, dtype=np.float32)
    else:
        normalized = (depth.astype(np.float32) - min_depth) / (max_depth - min_depth)
        normalized = np.clip(normalized, 0.0, 1.0)

    # Set background (-1) to 0 (black).
    normalized[depth == -1] = 0.0

    # Apply the colormap and return an array with shape (B, H, W, 4) and values in [0, 1].
    colored_rgba = colormap(normalized)

    # Drop the alpha channel and convert to uint8 in [0, 255].
    colored_rgb = (colored_rgba[..., :3] * 255).astype(np.uint8)

    return colored_rgb

class DISPMSELoss_vis(nn.Module):
    def __init__(self, args,l_list=None):
        """
        Args:
            depth_num (int): Number of input depth maps, from 1 to 3.
        """
        super(DISPMSELoss_vis, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.args = args
        self.l_list = l_list
        self.t_valid = 0.0001
        self.gamma = 0.9
        # Gradient loss module.
        self.lambda_weight=0.5

        # Register Sobel kernels.
        self.register_buffer('sobel_x', torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], dtype=torch.float32))
        self.register_buffer('sobel_y', torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], dtype=torch.float32))

    def compute_grad_loss(self, pred, gt, mask):
        """
        Improved sparse-ground-truth gradient loss.
        """
        # Compute prediction gradients.
        pred_grad_x = F.conv2d(pred, self.sobel_x.to(pred.device), padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y.to(pred.device), padding=1)

        # Compute target gradients only in valid regions.
        with torch.no_grad():
            # Dilate the mask to include neighborhood information.
            dilated_mask = F.max_pool2d(mask.float(), 3, stride=1, padding=1)
            valid_gt = gt * dilated_mask

            gt_grad_x = F.conv2d(valid_gt, self.sobel_x.to(pred.device), padding=1) * dilated_mask
            gt_grad_y = F.conv2d(valid_gt, self.sobel_y.to(pred.device), padding=1) * dilated_mask

        # Compute loss only within the original mask.
        loss_x = torch.abs(pred_grad_x - gt_grad_x) * mask
        loss_y = torch.abs(pred_grad_y - gt_grad_y) * mask

        return (loss_x + loss_y).mean()

    def forward(self, *args,depth_predictions_down=None,disp=None,f=None,vis=False,val=False,
                rgb_left=None,rgb_right=None,sparse=None,agg_preds=None):
        """
        Accept a variable number of depth inputs according to depth_num.
        """
        target = args[-1]
        depths = args[0]  # Get depth tensors.
        if isinstance(depths[0],list):
            depths = depths[0]
            depths = [depth.squeeze(-1).squeeze(1) for depth in depths]
        else:
            depths = [depth.squeeze(-1).squeeze(1) for depth in depths]  # Remove singleton dimensions.

        if disp is not None:
            if isinstance(disp[0],list):
                disp = disp[0]
                disp = [d.squeeze(-1).squeeze(1) for d in disp]
            else:
                disp = [d.squeeze(-1).squeeze(1) for d in disp]  # Remove singleton dimensions.

        assert all(depth.dim() == target.dim() for depth in depths), "inconsistent dimensions"

        if vis:
            self.visualize_pred_and_target(*depths, target=target,val=val,
                                           rgb_left=rgb_left,rgb_right=rgb_right,sparse=sparse)
            # visualize_all(self.args,depths[-1], target,save_path='tight_compare.png',
            #                    rgb_left=rgb_left,rgb_right=rgb_right,
            #                    sparse=sparse)
        gt = torch.clamp(target, min=0, max=self.args.z_max)
        seq_depth = [torch.clamp(pred, min=0, max=self.args.z_max) for pred in depths]
        if disp is not None:
            seq_disp = [torch.clamp(d, min=0, max=192) for d in disp]
        gt_down = F.interpolate(gt.unsqueeze(1),scale_factor=0.25,mode='nearest')
        mask_down = (gt_down > self.t_valid).float()
        mask = (gt > self.t_valid).type_as(seq_depth[0]).detach()
        num_valid = torch.sum(mask.unsqueeze(1), dim=[1, 2, 3])
        num_valid_down = torch.sum(mask_down.unsqueeze(1), dim=[1, 2, 3])
        n_predictions = len(seq_depth)
        loss = 0.0
        loss_grad=0.0
        loss_down=0.0
        if disp is not None:
            disp_gt=target.clone()
            b,h,w = target.shape
            disp_gt[target>0]=(f.view(-1, 1)*self.args.baseline/(target.reshape(b,-1)+1e-6))[target.reshape(b,-1)>0]

        # if agg_preds is not None:
        #     agg_loss=0.0
        #     mask0 = ((gt > 0) & (gt < self.args.z_max//4))
        #     mask1 = ((gt > 0) & (gt < self.args.z_max//2))
        #     mask = ((gt > 0) & (gt < self.args.z_max))
        #     agg_loss += 0.7 * F.smooth_l1_loss(agg_preds[0][mask0.bool()], gt[mask0.bool()], reduction='mean')
        #     agg_loss += 0.5 * F.smooth_l1_loss(agg_preds[1][mask1.bool()], gt[mask1.bool()], reduction='mean')
        #     agg_loss += 0.2 * F.smooth_l1_loss(agg_preds[2][mask.bool()], gt[mask.bool()], reduction='mean')
        #     loss += 1 * agg_loss
        # if self.args.update_with == "selective" and self.args.attention_active:
        #     agg_loss += 0.7 * F.smooth_l1_loss(seq_disp[-1][mask0.bool()], gt[mask0.bool()], reduction='mean')
        #     agg_loss += 0.5 * F.smooth_l1_loss(seq_disp[-1][mask1.bool()], gt[mask1.bool()], reduction='mean')
        #     agg_loss += 0.2 * F.smooth_l1_loss(seq_disp[-1][mask.bool()], gt[mask.bool()], reduction='mean')
        #     loss += 1 * agg_loss
        for i in range(n_predictions):
            i_weight = self.gamma ** ((n_predictions - 1) - i)
            if i == 0 and n_predictions!=1:
                i_weight=0.2
                
            loss_type = getattr(self.args, "loss_type", ["l2"])
            loss_source = getattr(self.args, "loss_source", ["depth"])
            
            if 'l1' in loss_type:
                if 'depth' in loss_source:
                    i_loss_depth1 = torch.abs(seq_depth[i] - gt) * mask
                    i_loss_depth1 = torch.sum(i_loss_depth1.unsqueeze(1), dim=[1, 2, 3]) / (num_valid + 1e-8)
                    loss += i_weight * i_loss_depth1.sum()
                    if depth_predictions_down is not None:
                        i_loss_down = torch.abs(depth_predictions_down[i] - gt_down) * mask_down
                        i_loss_down = torch.sum(i_loss_down.unsqueeze(1), dim=[1, 2, 3]) / (num_valid_down + 1e-8)
                        loss_down += i_weight * i_loss_down.sum()
                if disp is not None and 'disp' in loss_source:
                    i_loss_disp1 = torch.abs(seq_disp[i] - disp_gt) * mask
                    i_loss_disp1 = torch.sum(i_loss_disp1.unsqueeze(1), dim=[1, 2, 3]) / (num_valid + 1e-8)
                    loss += i_weight * i_loss_disp1.sum()
            if 'l2' in loss_type:
                if 'depth' in loss_source:
                    i_loss_depth2 = torch.pow(seq_depth[i] - gt, 2) * mask
                    i_loss_depth2 = torch.sum(i_loss_depth2.unsqueeze(1) , dim=[1, 2, 3]) / (num_valid + 1e-8)
                    loss += i_weight * i_loss_depth2.sum()
                    if depth_predictions_down is not None:
                        i_loss_down = torch.abs(depth_predictions_down[i] - gt_down) * mask_down
                        i_loss_down = torch.sum(i_loss_down.unsqueeze(1), dim=[1, 2, 3]) / (num_valid_down + 1e-8)
                        loss_down += i_weight * i_loss_down.sum()
                if disp is not None and 'disp' in loss_source:
                    i_loss_disp2 = torch.pow(seq_disp[i] - disp_gt, 2) * mask
                    i_loss_disp2 = torch.sum(i_loss_disp2.unsqueeze(1) , dim=[1, 2, 3]) / (num_valid + 1e-8)
                    loss += i_weight * i_loss_disp2.sum()
            if 'log' in loss_type:
                eps = 1e-6  # Avoid log(0).
                if 'depth' in loss_source:
                    # Depth log loss.
                    pred_depth = torch.clamp(seq_depth[i], min=eps)  # Ensure positive values.
                    log_diff = torch.log(pred_depth) - torch.log(gt + eps)
                    # Scale-invariant loss computation.
                    term1 = torch.mean((log_diff  **  2) * mask)  # Base log loss.
                    term2 = torch.mean(log_diff * mask)  **  2  # Shift penalty term.
                    i_loss_log_depth = term1 - self.lambda_weight * term2
                    loss += i_weight * i_loss_log_depth
                if disp is not None and 'disp' in loss_source:
                    # Optional disparity log loss.
                    pred_disp = torch.clamp(seq_disp[i], min=eps)
                    log_diff_disp = torch.log(pred_disp) - torch.log(disp_gt + eps)

                    term1_disp = torch.mean((log_diff_disp  **  2) * mask)
                    term2_disp = torch.mean(log_diff_disp * mask)  **  2
                    i_loss_log_disp = term1_disp - getattr(self.args, "lambda_weight", 0.5) * term2_disp

                    loss += i_weight * i_loss_log_disp
            if 'grad' in loss_type:
                grad_loss_depth = self.compute_grad_loss(
                    seq_depth[i].unsqueeze(1),
                    gt.unsqueeze(1),
                    mask.unsqueeze(1)
                )
                loss_grad += i_weight * grad_loss_depth
                # Disparity gradient loss.
                if disp is not None and 'disp' in loss_source:
                    grad_loss_disp = self.compute_grad_loss(
                        seq_disp[i].unsqueeze(1),
                        disp_gt.unsqueeze(1),
                        mask.unsqueeze(1)
                    )
                    loss_grad += i_weight * grad_loss_disp * getattr(self.args, "grad_weight", 1.0)
            if 'depth' in loss_source and 'disp' in loss_source:
                loss /= 2
        if 'grad' in loss_type:
            return loss*0.8+loss_grad*0.2
        else:
            return loss+loss_down*0.5

    def visualize_pred_and_target(self, *depths, target,
                                  save_path='predict_gt_compare.png',
                                  val=False,rgb_left=None,rgb_right=None,
                                  sparse=None):
        """
        Visualize a variable number of depth maps according to depth_num.
        """
        # Use the first batch element for depth and target maps.
        target_img = target[0, :, :]
        # target_img[target_img>0]=(f[0]*self.args.baseline/(target_img+1e-6))[target[0]>0]

        depth_imgs = [depth[0, :, :] for depth in depths]
        
        # Limit depth visualizations to avoid excessive plots.
        if val:
            if len(depth_imgs) > 1 and len(depth_imgs) < 4:
                depth_imgs = depth_imgs[-2:] # Keep the last two maps.
            elif len(depth_imgs) > 4:
                # When a full sequence is provided, retain the final len(depths) maps or all maps as required.
                # Use len(depth_imgs) for consistency.
                pass
            else:
                depth_imgs = [depth_imgs[-1]]
                
        depth_num = len(depth_imgs)

        # Visualize each depth tensor and the target map.
        if rgb_left is not None:
            fig, axs = plt.subplots(depth_num + 2, 2, figsize=(12, 3 * (depth_num + 2)))
        else:
            fig, axs = plt.subplots(depth_num + 1, 2, figsize=(12, 3 * (depth_num + 1)))

        if rgb_left is not None and rgb_right is not None:
            # Visualize RGB images in the first two columns for the first row
            # Convert to a displayable RGB image.
            rgb_left_img = normalize_for_display(rgb_left.cpu().numpy()[0].transpose(1, 2, 0)).astype(np.uint8)
            rgb_right_img = normalize_for_display(rgb_right.cpu().numpy()[0].transpose(1, 2, 0)).astype(np.uint8)

            axs[0, 0].imshow(rgb_left_img)
            axs[0, 0].set_title('RGB Left')
            axs[0, 0].axis('off')
            # cv2.imwrite('rgb_left_image.png', rgb_left.cpu().numpy()[0].transpose(1, 2, 0))
            # cv2.imwrite('rgb_right_image.png', rgb_right.cpu().numpy()[0].transpose(1, 2, 0))
            axs[0, 1].imshow(rgb_right_img)
            axs[0, 1].set_title('RGB Right')
            axs[0, 1].axis('off')

            if sparse is not None:
                # Create a mask where depth > 0 is true.
                depth_mask = (sparse[0].squeeze(0) > 0).cpu().numpy()
                # Overlay highlighted regions on rgb_left_img.

                # Use a gradient colormap for the depth mask.
                norm = plt.Normalize(vmin=0, vmax=sparse[0].squeeze(0).max())  # Display range.
                cmap = plt.get_cmap('rainbow')  # Select the gradient colormap.

                # Create an RGB overlay where the colormap is applied only to depth_mask.
                gradient_overlay = cmap(norm(sparse[0].squeeze(0).cpu().numpy()))   # Get gradient colors.
                gradient_overlay = (gradient_overlay[:, :, :3] * 255).astype(np.uint8)  # Extract RGB channels and convert to uint8.

                # Overlay the gradient colors on the original RGB image.
                rgb_left_img[depth_mask] = gradient_overlay[depth_mask]

                # Render the overlaid left RGB image.
                axs[0, 0].imshow(rgb_left_img)
                axs[0, 0].set_title('RGB Left with Depth > 0')
                axs[0, 0].axis('off')
            # Visualize depth maps starting from the second row
            for i, depth_img in enumerate(depth_imgs):
                depth_colored = apply_colormap_to_depth(depth_img, stretch_range=(0, self.args.z_max)).squeeze(0)
                axs[i + 1, 0].imshow(depth_colored)
                axs[i + 1, 0].set_title(f'Depth {i + 1} Prediction')
                axs[i + 1, 0].axis('off')

                # Calculate error map with respect to target
                error_map, cmap, norm = apply_log_error_map(depth_img, target_img)
                error_display = axs[i + 1, 1].imshow(error_map, cmap=cmap, norm=norm)
                cbar = plt.colorbar(error_display, ax=axs[i + 1, 1])
                cbar.set_label('Relative Error (%)')
                cbar.set_ticks([0, 0.01, 0.02, 0.03, 0.04, 0.05, float('inf')])
                cbar.set_ticklabels(['0%', '<1%', '1%-2%', '2%-3%', '3%-4%', '4%-5%', '>5%'])
                rmse = RMSE(depth_img, target_img)
                axs[i + 1, 1].set_title(f'Depth {i + 1} Error Map (RMSE: {rmse:.2f} mm)')
                axs[i + 1, 1].axis('off')
        else:
            # Iterate through depth maps for visualization.
            for i, depth_img in enumerate(depth_imgs):
                depth_colored = apply_colormap_to_depth(depth_img, stretch_range=(0, self.args.z_max)).squeeze(0)
                axs[i, 0].imshow(depth_colored)
                axs[i, 0].set_title(f'Depth {i + 1} Prediction')
                axs[i, 0].axis('off')

                # Compute the error relative to the target map.
                error_map, cmap, norm = apply_log_error_map(depth_img, target_img)
                error_display = axs[i, 1].imshow(error_map, cmap=cmap, norm=norm)
                cbar = plt.colorbar(error_display, ax=axs[i, 1])
                cbar.set_label('Relative Error (%)')
                cbar.set_ticks([0, 0.01, 0.02, 0.03, 0.04, 0.05, float('inf')])
                cbar.set_ticklabels(['0%', '<1%', '1%-2%', '2%-3%', '3%-4%', '4%-5%', '>5%'])
                rmse = RMSE(depth_img, target_img)
                axs[i, 1].set_title(f'Depth {i + 1} Error Map (RMSE: {rmse:.2f} mm)')
                axs[i, 1].axis('off')

        if rgb_right is not None and rgb_left is not None:
            last_depth_num=depth_num+1
        else:
            last_depth_num=depth_num

        # Visualize the target depth map.
        target_colored = apply_colormap_to_depth(target_img, stretch_range=(0, self.args.z_max)).squeeze()
        axs[last_depth_num, 0].imshow(target_colored)
        axs[last_depth_num, 0].set_title('Target Depth')
        axs[last_depth_num, 0].axis('off')

        # Visualize the error between the final and initial depth maps.
        if last_depth_num > 1:
            error_map, cmap, norm = apply_log_error_map(depth_imgs[0].float(), depth_imgs[-1].float())
            error_display = axs[last_depth_num, 1].imshow(error_map, cmap=cmap, norm=norm)
            cbar = plt.colorbar(error_display, ax=axs[last_depth_num, 1])
            cbar.set_label('Relative Error (%)')
            cbar.set_ticks([0, 0.01, 0.02, 0.03, 0.04, 0.05, float('inf')])
            cbar.set_ticklabels(['0%', '<1%', '1%-2%', '2%-3%', '3%-4%', '4%-5%', '>5%'])
            rmse = RMSE(depth_imgs[0], depth_imgs[-1])
            axs[last_depth_num, 1].set_title(f'Depth 1 vs Depth {last_depth_num} Error Map (RMSE: {rmse:.2f} mm)')
            axs[last_depth_num, 1].axis('off')
        else:
            # Remove the last unused subplot.
            fig.delaxes(axs[last_depth_num, 1])

        # Save the figure.
        plt.tight_layout()
        plt.savefig(save_path, dpi=250)
        # swanlab.log({"vis_result": swanlab.Image(plt)})
        plt.close(fig)

