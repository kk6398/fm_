#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from ..scene.cameras import Camera
import numpy as np
from .general_utils import PILtoTorch
from .graphics_utils import fov2focal
from ..scene.dataset_readers import CameraInfo

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size
    resolution_args = -1
    if resolution_args in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * resolution_args)), round(orig_h/(resolution_scale * resolution_args))
    else:  # should be a type that converts to float
        if resolution_args == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / resolution_args

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))     # 1600，1200

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)  # 3x1200x1600      # 将original图像resize为裁剪尺寸，并读取

    gt_image = resized_image_rgb[:3, ...]  # start被省略了，表示从数组的起始位置开始切片，stop被设置为3，表示切片结束位置为索引为2的元素（不包含在切片中），step也被省略了，表示使用默认的步长1。
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device="cuda")

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []      # validation_configs时，先读取所有相机视角的参数，再选取5个

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_from_camInfos_selection(id, cam_infos, resolution_scale, args):
    # 只有一个视角的相机参数
    return loadCam(args, id, cam_infos[id], resolution_scale)

# def camera_to_JSON(id, camera : Camera):
def camera_to_JSON(id, camera : CameraInfo):
    
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
