import torch
import torch.nn.functional as F
def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)

def context_upsample(disp_low, up_weights, up_factor=4):
    # Upsample low-resolution depth with learned convex weights.

    b, c, h, w = disp_low.shape

    disp_unfold = F.unfold(disp_low.reshape(b, c, h, w), 3, 1, 1).reshape(b, -1, h, w)
    disp_unfold = F.interpolate(disp_unfold, (h * up_factor, w * up_factor),
                                mode='nearest').reshape(b, 9, h * up_factor, w * up_factor)
    disp_nei = (disp_unfold * up_weights).sum(dim=1, keepdim=True)
    return disp_nei.squeeze(1)


def soft_code(volume, points_d, d_down_index, mask_d, z_max, D_all):
    """
    volume:      initialized left/right feature volume with shape (B, C_chan, H', W', D_all)
    points_d:    (B, N, 3)
    d_down_index:(B, N, 2)  Downsampled pixel indices.
    mask_d:      (B, N)     Valid point mask.
    z_max:       float
    D_all:       int
    """
    B, C_chan, H, W, D = volume.shape
    device, dtype = volume.device, volume.dtype

    # 1) Compute continuous depth indices and interpolation weights.
    Z = points_d[..., 2]  # (B, N)
    d_cont = (Z / z_max) * D_all  # (B, N)
    k0 = torch.floor(d_cont).long().clamp(min=0, max=D_all-1)   # (B, N)
    k1 = (k0 + 1).clamp(max=D_all-1)                            # (B, N)
    w1 = (d_cont - k0.float()).clamp(min=0.0, max=1.0).to(dtype=dtype)  # (B, N) Match the volume dtype.
    w0 = (1.0 - w1).to(dtype=dtype)                             # (B, N) Match the volume dtype.

    # Keep points with mask_d == 1 and 0 < Z < z_max.
    valid = mask_d.to(torch.bool) & (Z>0) & (Z<z_max)  # (B, N)

    # 3) Flatten all valid coordinates and interpolation weights.
    #    b_idxs: (M,),  h_idxs: (M,), w_idxs: (M,), k0_flat: (M,), k1_flat: (M,), w0_flat: (M,), w1_flat: (M,)
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, Z.shape[1])  # (B, N)
    b_flat   = b_idx[valid]
    h_flat   = d_down_index[..., 0][valid].to(device)
    w_flat   = d_down_index[..., 1][valid].to(device)
    k0_flat  = k0[valid]
    k1_flat  = k1[valid]
    w0_flat  = w0[valid]
    w1_flat  = w1[valid]

    # 4) Sparse-depth channel index.
    depth_ch = C_chan - 1

    # Accumulate values with index_put_.
    idx0 = (b_flat,
            torch.full_like(b_flat, depth_ch),
            h_flat, w_flat, k0_flat)
    idx1 = (b_flat,
            torch.full_like(b_flat, depth_ch),
            h_flat, w_flat, k1_flat)

    volume.index_put_(idx0, w0_flat, accumulate=True)
    volume.index_put_(idx1, w1_flat, accumulate=True)

    return volume


def depth_regression(prob,max_depth,interval=1):
    prob=prob.squeeze(1)
    depth_values = torch.arange(0, max_depth, interval, dtype=prob.dtype, device=prob.device)
    depth_values = depth_values.view(1, int(max_depth/interval), 1, 1)
    out = torch.sum(prob*depth_values,1, keepdim=True)

    return out


def project_left_to_right_with_crop_batch(left_depth_map, baseline, f,
                                          down_sample=False, down_factor=1):
    '''
    :param left_depth_map: Expected shape is (H, W).
    :param baseline:
    :param f:
    :param down_sample:
    :param down_factor:
    :return:
    '''
    # Get the batch size B and pixel count N for each image.
    B, N, _ = left_depth_map.shape

    # Use left- and right-view camera intrinsics K_02 and K_03.
    if not down_sample:
        down_factor = 1

    # u ->352 horizontal coordinate, v->1216vertical coordinate
    # Use left-view pixel coordinates (u, v) and depth d.
    v = left_depth_map[:, :, 1]  # (B, N)
    d = left_depth_map[:, :, 2]  # (B, N)
    # Back-project to 3D camera coordinates. (X, Y, Z)
    X = left_depth_map[:, :, 0]
    Y = v - (baseline * f.unsqueeze(1) / (d * down_factor + 1e-4))

    # Create output coordinates with shape (B, N, 3): right-view pixels (u', v') and depth d.
    right_view_coordinates_down = torch.stack((X, Y, d), dim=-1)  # (B, N, 3)
    return right_view_coordinates_down
from matplotlib import pyplot as plt
import numpy as np

def groupwise_correlation(fea1, fea2, num_groups):
    C = fea1.size(1)
    group_size = C // num_groups

    # Reshape to [N, num_groups, group_size].
    fea1_grouped = fea1.view(-1, num_groups, group_size)
    fea2_grouped = fea2.view(-1, num_groups, group_size)

    # Compute groupwise correlation with element-wise products.
    return torch.einsum('ngs,ngs->ng', fea1_grouped, fea2_grouped) / group_size

def vis_project(volume_orig, volume_change,depth=25):
    #Default depth slice for visualization.
    b = 2
    i = depth
    C = 3
    slice_orig = volume_orig[b, C:2 * C,i].permute(1, 2, 0).cpu().numpy()  # (H,W,3)
    slice_change = volume_change[b, C:2 * C, i].permute(1, 2, 0).cpu().numpy()  # (H,W,3)

    def normalize_to_uint8(img):
        # img: H×W×3 float
        lo, hi = img.min(), img.max()
        if hi <= lo:
            return np.zeros_like(img, dtype=np.uint8)
        norm = (img - lo) / (hi - lo)  # Normalize to [0, 1].
        out = (norm * 255.0).clip(0, 255).astype(np.uint8)
        return out

    slice_orig_u8 = normalize_to_uint8(slice_orig)
    slice_change_u8 = normalize_to_uint8(slice_change)
    # 5) Visual comparison.
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for a, img, title in zip(ax,
                             [slice_orig_u8, slice_change_u8],
                             ['Original_Shift', 'New_Shift']):
        a.imshow(img)  # Matplotlib interprets uint8 values as 0-255 RGB.
        a.set_title(title)
        a.axis('off')
        for spine in a.spines.values():
            spine.set_visible(True)  # Show the axis spine.
            spine.set_linewidth(2)  # Set the line width.
            spine.set_edgecolor('red')  # Highlight the comparison border.
    plt.title(f"Shifted on Depth={i}")
    plt.tight_layout()
    plt.savefig(f'depth={i}_shifted_compare_uint8.png', dpi=350)


def vis_overlay(volume_orig, volume_change, depth=25, alpha=0.5, mode='blend'):
    """
    Overlay the original and updated projections at the selected depth plane.

    Args:
      volume_orig:   Tensor or ndarray (B, 2C, D, H, W)
      volume_change: same as volume_orig
      depth:         int, depth plane to display
      alpha:         float in [0, 1], blend ratio used only when mode='blend'
      mode:          'blend' or 'anaglyph'
                     - 'blend'   : simple alpha blending
                     - 'anaglyph': red channel from the original projection and green/blue channels from the updated projection
    """
    b = 2
    C = 3
    # Extract an H x W x 3 uint8 RGB image.
    def get_u8(vol):
        img = vol[b, C:2 * C, depth].transpose(1, 2, 0)
        lo, hi = img.min(), img.max()
        if hi <= lo: return np.zeros_like(img, np.uint8)
        norm = (img - lo) / (hi - lo)
        return (norm * 255).clip(0, 255).astype(np.uint8)

    img_o = get_u8(volume_orig.cpu().numpy())
    img_c = get_u8(volume_change.cpu().numpy())

    if mode == 'blend':
        over = (img_o.astype(np.float32) * (1 - alpha) + img_c.astype(np.float32) * alpha)
        over = over.clip(0, 255).astype(np.uint8)
    else:  # anaglyph
        over = np.zeros_like(img_o)
        over[..., 0] = img_o[..., 0]
        over[..., 1:] = img_c[..., 1:]

    # Display the overlay image.
    plt.figure(figsize=(5, 5))
    plt.imshow(over)
    plt.title(f"{'Blend' if mode == 'blend' else 'Anaglyph'} @Depth={depth}\n(alpha={alpha})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'depth={depth}_overlay_uint8.png', dpi=350)
