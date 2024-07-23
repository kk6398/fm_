import shutil
from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
# from scipy.spatial.transform import Rotation as R
from torch import Tensor
# #
# from ..misc.cropping import center_crop_intrinsics
# from ..model.model import ModelExports
# from ..model.projection import homogenize_points, sample_image_grid, unproject
from ..third_party.colmap.read_write_model import Camera, Image, read_model, write_model
# from open3d.utility import Quaternion
from scipy.spatial.transform import Rotation as R



def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret




def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    # pyre-ignore [16]: `torch.Tensor` has no attribute `new_tensor`.
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(*batch_dim, 4)




def flowmap_2_gs(intrinsics, extrinsics, frame_paths, batch_video):
        
    cameras = {}
    images = {}
    
    camera_id = 1
    model_id = 1
    model_name = "PINHOLE"
    width = batch_video.shape[-1]    # 4032
    height = batch_video.shape[-2]   # 3024
    # print("batch_video.shape: ", batch_video.shape)        # torch.Size([1, 20, 3, 3024, 4032])

    # params = np.array([intrinsics[0, 0, 0, 0], intrinsics[0, 0, 1, 1], intrinsics[0, 0, 0, 2], intrinsics[0, 0, 1, 2]])
    # print("intrinsics: ", intrinsics)                # 归一化后的参数
    params_cpu = intrinsics.cpu().detach().numpy()
    params = [params_cpu[0, 0, 0, 0], params_cpu[0, 0, 1, 1], params_cpu[0, 0, 0, 2], params_cpu[0, 0, 1, 2]]
    # print("params: ", params)               # params:   [0.7121296, 0.99698144, 0.5, 0.5]
    cameras[camera_id] = Camera(id=camera_id,                    # 内参信息
                                    model=model_name,
                                    width=width,
                                    height=height,
                                    params=params)        
    # print("cameras: ", cameras)
    
    for i in range(extrinsics.shape[1]):
        extrinsic_matrix = extrinsics[:, i, :, :]     # 外参也初始化了为单位矩阵 #
        extrinsic_matrix = extrinsic_matrix.squeeze(0)
        # print("extrinsic_matrix: ", extrinsic_matrix.shape)   # torch.Size([4, 4])
        image_id = i + 1
        R = extrinsic_matrix[:3, :3]      # 3*3的单位矩阵
        t = extrinsic_matrix[:3, 3]       # 1*3的零矩阵
        qvec = matrix_to_quaternion(R).tolist()    # 转换为四元数  [1.0, 0.0, 0.0, 0.0]
        tvec = t
        camera_id = 1            
        image_name = frame_paths[i].name     # 1: PosixPath('/data2/hkk/datasets/flowmap/llff_fern/IMG_4026.JPG')
        
        images[image_id] = Image(                               # 外参信息
            id=image_id, qvec=qvec, tvec=tvec,
            camera_id=camera_id, name=image_name, xys=[], point3D_ids=[])

    return images, cameras           # 分别对应  外参和内参