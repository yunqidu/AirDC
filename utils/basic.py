import numpy as np
def reshape_to_BHW(tensor):
    # Remove the singleton channel dimension for tensors shaped [B, 1, H, W].
    if tensor.dim() == 4 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    return tensor


def pix2cam(depth, K, crop_info=None, return_idx=False,return_pix=False):
    depth=depth.copy()
    K=K.copy()
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

    # Adjust pixel coordinates when crop metadata is available.
    if crop_info is not None:
        crop_top = crop_info.get('crop_top', 0)
        crop_left = crop_info.get('crop_left', 0)
        mask = depth > 0  # Create a mask for valid positive depth values.
        z = depth[mask]
        i = i[mask]
        j = j[mask]

        # Back-project depth to 3D points.
        x = (i+ crop_left - K[0, 2]) * z / K[0, 0]
        y = -(j+ crop_top - K[1, 2]) * z / K[1, 1]  # Invert the y-axis direction.
        points = np.stack((x, y, z), axis=-1)
    else:
        mask = depth > 0  # Create a mask for valid positive depth values.
        z = depth[mask]
        i = i[mask]
        j = j[mask]

        # Back-project depth to 3D points.
        x = (i - K[0, 2]) * z / K[0, 0]
        y = -(j - K[1, 2]) * z / K[1, 1]  # Invert the y-axis direction.
        points = np.stack((x, y, z), axis=-1)
    # Return pixel indices or point clouds.
    if return_idx:
        return points, np.stack((j, i), axis=-1)
    elif return_pix:
        # Prepare the final result: u, v, d, x, y, z
        depth_coord_d = np.column_stack((i, j, z, x, y, z))
        return points,depth_coord_d
    else:
        return points
