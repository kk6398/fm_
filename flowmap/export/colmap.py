import shutil
from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor
# #
from ..misc.cropping import center_crop_intrinsics
from ..model.model import ModelExports
from ..model.projection import homogenize_points, sample_image_grid, unproject
from ..third_party.colmap.read_write_model import Camera, Image, read_model, write_model
# from open3d.utility import Quaternion
from scipy.spatial.transform import Rotation as R


def read_ply(path: Path) -> tuple[
    Float[np.ndarray, "point 3"],  # xyz
    Float[np.ndarray, "point 3"],  # rgb
]:
    # Adapted from https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/dataset_readers.py#L107
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    rgb = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    return xyz, rgb


def write_ply(
    path: Path,
    xyz: Float[np.ndarray, "point 3"],
    rgb: Float[np.ndarray, "point 3"],
) -> None:
    # Adapted from https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/dataset_readers.py#L115
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb * 255), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def export_to_colmap_e2e(
    colors: Float[Tensor, "batch frame 3 height width"],
    depths: Float[Tensor, "batch frame height width"],
    intrinsics_input: Float[Tensor, "batch frame 3 3"],
    extrinsics_input: Float[Tensor, "batch frame 4 4"],
    frame_paths: list[Path],                             # 怎么获得    都是overfit.py的变量
    uncropped_exports_shape: tuple[int, int],            # 怎么获得
    uncropped_videos: Float[Tensor, "batch frame 3 uncropped_height uncropped_width"],    # 怎么获得
    path: Path,                                         # 怎么获得
) -> None:
    _, _, h_cropped, w_cropped = depths.shape
    h_uncropped, w_uncropped = uncropped_exports_shape
    intrinsics_output = center_crop_intrinsics(
        intrinsics_input,
        (h_cropped, w_cropped),
        (h_uncropped, w_uncropped),
    )
    print("intrinsics_output: ", intrinsics_output.shape)   # torch.Size([1, 20, 3, 3])
    print("intrinsics_input: ", intrinsics_input.shape)     # torch.Size([1, 20, 3, 3])
    print("extrinsics_input: ", extrinsics_input.shape)     # torch.Size([1, 20, 4, 4])
    # Write out the camera parameters.
    # print("1111111111111111111111")
    # print("path: ", path)
    sparse_path = path / "sparse/0"
    _, _, _, h_full, w_full = uncropped_videos.shape
    write_colmap_model(
        sparse_path,
        extrinsics_input[0],
        intrinsics_output[0],
        [path.name for path in frame_paths],
        (h_full, w_full),
    )
    print("extrinsics_input[0].shape: ", extrinsics_input[0].shape)     # torch.Size([20, 4, 4])
    print("intrinsics_output[0].shape: ", intrinsics_output[0].shape)   # torch.Size([20, 3, 3])

    # Define the point cloud. For compatibility with 3D Gaussian Splatting, this is
    # stored as a .ply instead of Points3D, which seems to be intended for a much
    # smaller number of points.
    _, _, dh, dw = depths.shape
    xy, _ = sample_image_grid((dh, dw), extrinsics_input.device)
    bundle = zip(
        extrinsics_input[0],
        intrinsics_output[0],
        depths[0],
        colors[0],
    )
    points = []
    colors = []
    for extrinsics, intrinsics, depths, rgb in bundle:
        xyz = unproject(xy, depths, intrinsics)
        xyz = homogenize_points(xyz)
        xyz = einsum(extrinsics, xyz, "i j, ... j -> ... i")[..., :3]
        points.append(rearrange(xyz, "h w xyz -> (h w) xyz").detach().cpu().numpy())
        colors.append(rearrange(rgb, "c h w -> (h w) c").detach().cpu().numpy())
    points = np.concatenate(points)
    colors = np.concatenate(colors)

    sparse_path.mkdir(parents=True, exist_ok=True)
    write_ply(sparse_path / "points3D.ply", points, colors)

    # Write out the images.
    (path / "images").mkdir(exist_ok=True, parents=True)
    for frame_path in frame_paths:
        shutil.copy(frame_path, path / "images" / frame_path.name)



# 根据给定的输入，输出 colmap形式的文件: images sparse/0/cameras.bin sparse/0/images.bin sparse/0/points3D.ply
# 内参：经过了2步尺度的还原: 1 croped: [160,224]→[180,240] 2 反归一化--original video shape: [180,240]→[3024, 4032]
# 外参：将c2w 转化成 w2c的四元数
# 3D点：将深度、内参、外参、颜色 zip打包，将2D点转换到3D点，并根据3D points和colors输出points3D.ply文件
def export_to_colmap(              
    exports: ModelExports,         # 外参、内参、颜色、深度
    frame_paths: list[Path],       # 帧路径
    uncropped_exports_shape: tuple[int, int],
    uncropped_videos: Float[Tensor, "batch frame 3 uncropped_height uncropped_width"],
    path: Path,
) -> None:
    # Account for the cropping that FlowMap does during optimization.
    _, _, h_cropped, w_cropped = exports.depths.shape           # 160  224
    # print("-------------------------")
    # print('exports.depths.shape:',exports.depths.shape)  #      ([1, 20, 160, 224])
    # print('exports.intrinsics.shape:',exports.intrinsics.shape)  #   ([1, 20, 3, 3])
    # print('exports.extrinsics.shape:',exports.extrinsics.shape)  #   ([1, 20, 4, 4])
    # print('exports.colors.shape:',exports.colors.shape)  #   ([1, 20, 3, 160, 224])
    
    h_uncropped, w_uncropped = uncropped_exports_shape          # 180 240
    # print('uncropped_exports_shape', uncropped_exports_shape)  #   应该与pre_crop_shape一致    (180, 240)
    
    intrinsics = center_crop_intrinsics(
        exports.intrinsics,
        (h_cropped, w_cropped),         # 160  224
        (h_uncropped, w_uncropped),     # (180, 240)
    )
    # print("-------------------------")
    # print('crop_intrinsics  exports.intrinsics.shape:', intrinsics.shape)  #  ([1, 20, 3, 3])  # 待定：尺度没变，值乘相应倍数
    
    # Write out the camera parameters.      输出图像对应的内参、外参，为colmap做准备
    sparse_path = path / "sparse/0"
    # print('sparse_path:',sparse_path)  #  outputs/local/colmap/sparse/0
    
    _, _, _, h_full, w_full = uncropped_videos.shape         # 原视频帧的尺度
    # print('uncropped_videos.shape:',uncropped_videos.shape)  #   torch.Size([1, 20, 3, 3024, 4032])
    
    write_colmap_model(                           # 输出 cameras、 images.text/.bin文件
        sparse_path,           
        exports.extrinsics[0],
        intrinsics[0],
        [path.name for path in frame_paths],
        (h_full, w_full),                      # [3024, 4032]
    )

    # Define the point cloud. For compatibility with 3D Gaussian Splatting, this is
    # stored as a .ply instead of Points3D, which seems to be intended for a much
    # smaller number of points.
    _, _, dh, dw = exports.depths.shape               # ([1, 20, 160, 224])
    xy, _ = sample_image_grid((dh, dw), exports.extrinsics.device)  # 生成图像网格的坐标，这些坐标用于后续的3D点云生成。
    bundle = zip(                     # zip函数将外参、内参、深度图像和颜色图像打包在一起，以便在循环中一起处理。
        exports.extrinsics[0],
        exports.intrinsics[0],
        exports.depths[0],
        exports.colors[0],
    )
    points = []        # 初始化两个空列表，用于存储转换后的3D点和对应的颜色。
    colors = []
    for extrinsics, intrinsics, depths, rgb in bundle:        # 循环遍历之前打包的数据。
        xyz = unproject(xy, depths, intrinsics)               # 使用unproject函数将图像坐标和深度值转换为3D空间中的点(相机坐标系)
        xyz = homogenize_points(xyz)                          # 将3D点同质化，即增加一个维度以便于矩阵乘法。
        xyz = einsum(extrinsics, xyz, "i j, ... j -> ... i")[..., :3]  # 将外参矩阵与同质化的3D点相乘，得到世界坐标系中的3D点，并去除同质化的维度。
        points.append(rearrange(xyz, "h w xyz -> (h w) xyz").detach().cpu().numpy())  # 将转换后的3D点和颜色分别添加到对应的列表中
        colors.append(rearrange(rgb, "c h w -> (h w) c").detach().cpu().numpy())
    points = np.concatenate(points)      # 将所有3D点和颜色合并成一个NumPy数组
    colors = np.concatenate(colors)

    sparse_path.mkdir(parents=True, exist_ok=True)
    write_ply(sparse_path / "points3D.ply", points, colors)   # 使用write_ply函数(from 3dgs)将3D点和颜色写入.ply文件

    # Write out the images.
    (path / "images").mkdir(exist_ok=True, parents=True)
    for frame_path in frame_paths:
        shutil.copy(frame_path, path / "images" / frame_path.name)


def read_colmap_model(
    path: Path,
    device: torch.device = torch.device("cpu"),
    reorder: bool = True,
) -> tuple[
    Float[Tensor, "frame 4 4"],  # extrinsics
    Float[Tensor, "frame 3 3"],  # intrinsics
    list[str],  # image names
]:
    model = read_model(path)
    if model is None:
        raise FileNotFoundError()
    cameras, images, _ = model

    all_extrinsics = []
    all_intrinsics = []
    all_image_names = []

    for image in images.values():
        camera: Camera = cameras[image.camera_id]

        # Read the camera intrinsics.
        intrinsics = torch.eye(3, dtype=torch.float32, device=device)
        if camera.model == "SIMPLE_PINHOLE":
            fx, cx, cy = camera.params
            fy = fx
        elif camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        intrinsics[0, 0] = fx
        intrinsics[1, 1] = fy
        intrinsics[0, 2] = cx
        intrinsics[1, 2] = cy
        intrinsics[0] /= camera.width
        intrinsics[1] /= camera.height
        all_intrinsics.append(intrinsics)

        # Read the camera extrinsics.
        qw, qx, qy, qz = image.qvec
        w2c = torch.eye(4, dtype=torch.float32, device=device)
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
        w2c[:3, :3] = torch.tensor(rotation, dtype=torch.float32, device=device)
        w2c[:3, 3] = torch.tensor(image.tvec, dtype=torch.float32, device=device)
        extrinsics = w2c.inverse()
        all_extrinsics.append(extrinsics)

        # Read the image name.
        all_image_names.append(image.name)

    # Since COLMAP shuffles the images, we generally want to re-order them according
    # to their file names so that they form a video again.
    if reorder:
        ordered = sorted([(name, index) for index, name in enumerate(all_image_names)])
        indices = torch.tensor([index for _, index in ordered])
        all_extrinsics = [all_extrinsics[index] for index in indices]
        all_intrinsics = [all_intrinsics[index] for index in indices]
        all_image_names = [all_image_names[index] for index in indices]

    return torch.stack(all_extrinsics), torch.stack(all_intrinsics), all_image_names


def write_colmap_model(
    path: Path,
    extrinsics: Float[Tensor, "frame 4 4"],           # exports.extrinsics[0],
    intrinsics: Float[Tensor, "frame 3 3"],           # intrinsics[0]
    image_names: list[str],                           # [path.name for path in frame_paths]
    image_shape: tuple[int, int],                     # [3024, 4032]
) -> None: 
    h, w = image_shape      # original video shape  [3024, 4032]
    print("=================================")
    print("intrinsics.shape: ", intrinsics.shape)  # torch.Size([20, 3, 3])
    print("extrinsics.shape: ", extrinsics.shape)  # torch.Size([20, 4, 4])
    # print("intrinsics: ", intrinsics)
    # print("extrinsics: ", extrinsics)
    print("=================================")
    
    # Define the cameras (intrinsics).
    cameras = {} 
    for index, k in enumerate(intrinsics):              # 按索引提取每张图像对应的内参
        id = index + 1   

        # Undo the normalization we apply to the intrinsics.
        k = k.detach().clone()
        k[0] *= w              # 将k矩阵的第一行的每个元素乘以变量w，第二行的每个元素乘以变量h
        k[1] *= h              # 为了撤销之前对内参进行的归一化处理

        # Extract the intrinsics' parameters.
        fx = k[0, 0]                         # fx和fy分别是沿着图像宽度和高度的焦距，cx和cy分别是图像的主点坐标
        fy = k[1, 1]
        cx = k[0, 2]
        cy = k[1, 2]

        cameras[id] = Camera(id, "PINHOLE", w, h, (fx, fy, cx, cy)) 
        # print("cameras[id]", cameras[id])
        # 创建了一个Camera对象，并将id、相机类型（这里是"PINHOLE"，即针孔相机模型）、
        # 图像的宽度和高度以及焦距和主点坐标作为参数传递给Camera类的构造函数
        # cameras[id] Camera(id=1, model='PINHOLE', width=4032, height=3024, 
        # params=(tensor(3457.6367, device='cuda:0'), tensor(3457.6365, device='cuda:0'), 
        # tensor(2016., device='cuda:0'), tensor(1512., device='cuda:0')))

    # Define the images (extrinsics and names).
    images = {}

    # zip将两个列表中的对应元素打包成一个个元组
    for index, (c2w, name) in enumerate(zip(extrinsics, image_names)):      # 按索引
        
        id = index + 1                                     

        # Convert the extrinsics to COLMAP's format.
        w2c = c2w.inverse().detach().cpu().numpy()                # 从世界坐标到相机坐标的转换
        # R.from_matrix方法从3x3旋转矩阵中获取四元数表示，然后使用as_quat方法将其转换为四元数形式（qw, qx, qy, qz）
        qx, qy, qz, qw = R.from_matrix(w2c[:3, :3]).as_quat()     # R: Rotation
        qvec = np.array((qw, qx, qy, qz))                         # 转换成数组
        tvec = w2c[:3, 3]                                         # 提取平移向量，即从世界坐标到相机坐标的转换矩阵的最后列
        images[id] = Image(id, qvec, tvec, id, name, [], [])

    path.mkdir(exist_ok=True, parents=True)
    write_model(cameras, images, None, path)           # 将相机参数、图像数据和3D点云数据写入，形成 cameras、 images.text/.bin文件
    
    
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







    # def camera_params(        
    #     colors: Float[Tensor, "batch frame 3 height width"],
    #     depths: Float[Tensor, "batch frame height width"],
    #     intrinsics_input: Float[Tensor, "batch frame 3 3"],
    #     extrinsics_input: Float[Tensor, "batch frame 4 4"],
    #     frame_paths: list[Path],                             # 怎么获得    都是overfit.py的变量
    #     uncropped_exports_shape: tuple[int, int],            # 怎么获得
    #     uncropped_videos: Float[Tensor, "batch frame 3 uncropped_height uncropped_width"],    # 怎么获得
    #     path: Path,                                         # 怎么获得
    # ) -> None:
        
            
    #     for index, k in enumerate(intrinsics):   # 遍历所有相机的内参
            
    #         h, w = image_shape      # original video shape  [3024, 4032]
    #         # Undo the normalization we apply to the intrinsics.
    #         k = k.detach().clone()
    #         k[0] *= w              # 将k矩阵的第一行的每个元素乘以变量w，第二行的每个元素乘以变量h
    #         k[1] *= h              # 为了撤销之前对内参进行的归一化处理

    #         # Extract the intrinsics' parameters.
    #         fx = k[0, 0]            # fx和fy分别是沿着图像宽度和高度的焦距，cx和cy分别是图像的主点坐标
    #         fy = k[1, 1]
    #         cx = k[0, 2]
    #         cy = k[1, 2]

    #         cameras[id] = Camera(id, "PINHOLE", w, h, (fx, fy, cx, cy)) 
    
    
    
    
    
  # 写一个函数，返回的是数据集所有图像的 viewpoints, 用于传入render函数
    # intrinsics [batch, frame, 3, 3]    torch.Size([1, 20, 3, 3])          # 归一化后的参数
    # extrinsics [batch, frame, 4, 4]    torch.Size([1, 20, 4, 4])
    # depth [C, N, H, W]       


# def flowmap_2_colmap(intrinsics, extrinsics, frame_paths, batch_video):
#
#     cameras = {}
#     images = {}
#
#     camera_id = 1
#     model_id = 1
#     model_name = "PINHOLE"
#     width = batch_video.shape[-1]    # 4032
#     height = batch_video.shape[-2]   # 3024
#     # print("batch_video.shape: ", batch_video.shape)        # torch.Size([1, 20, 3, 3024, 4032])
#     # print("width: ", width)
#     # print("height: ", height)
#
#     # params = np.array([intrinsics[0, 0, 0, 0], intrinsics[0, 0, 1, 1], intrinsics[0, 0, 0, 2], intrinsics[0, 0, 1, 2]])
#     # print("intrinsics: ", intrinsics)                # 归一化后的参数
#     params_cpu = intrinsics.cpu().detach().numpy()
#     params = [params_cpu[0, 0, 0, 0], params_cpu[0, 0, 1, 1], params_cpu[0, 0, 0, 2], params_cpu[0, 0, 1, 2]]
#     # print("params: ", params)               # params:   [0.7121296, 0.99698144, 0.5, 0.5]
#     cameras[camera_id] = Camera(id=camera_id,                    # 内参信息
#                                     model=model_name,
#                                     width=width,
#                                     height=height,
#                                     params=params)
#     # print("cameras: ", cameras)
#
#     for i in range(extrinsics.shape[1]):
#         extrinsic_matrix = extrinsics[:, i, :, :]     # 外参也初始化了为单位矩阵 #
#         extrinsic_matrix = extrinsic_matrix.squeeze(0)
#         # print("extrinsic_matrix: ", extrinsic_matrix.shape)   # torch.Size([4, 4])
#         image_id = i + 1
#         R = extrinsic_matrix[:3, :3]      # 3*3的单位矩阵
#         t = extrinsic_matrix[:3, 3]       # 1*3的零矩阵
#         # qvec = Quaternion(matrix=R).normalize().tolist()
#
#
#         qvec = matrix_to_quaternion(R).tolist()    # 转换为四元数  [1.0, 0.0, 0.0, 0.0]
#
#         # qvec = R.from_matrix(R).as_quat().normalize().tolist()
#         tvec = t
#         # print("qvec: ", qvec)
#         # print("tvec: ", tvec)
#         camera_id = 1
#         image_name = frame_paths[i].name     # 1: PosixPath('/data2/hkk/datasets/flowmap/llff_fern/IMG_4026.JPG')
#
#         images[image_id] = Image(                               # 外参信息
#             id=image_id, qvec=qvec, tvec=tvec,
#             camera_id=camera_id, name=image_name, xys=[], point3D_ids=[])
#     # print("images: ", images)
#
#     return images, cameras           # 外参和内参
#
