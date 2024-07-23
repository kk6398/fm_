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

import os
import random
import json
from ..utils.system_utils import searchForMaxIteration
from ..scene.dataset_readers import sceneLoadTypeCallbacks
from ..scene.gaussian_model import GaussianModel
from ..arguments import ModelParams, GroupParams
from ..utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from random import randint
from time import time


class Scene:            # 传入参数：dataset数据, gaussians模型
 
    gaussians : GaussianModel

    # def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
    def __init__(self, args : GroupParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        # print("entry the Scene class")
        # print("args: ", args)  # args:  <flowmap.arguments.ModelParams object at 0x7f8b1c0b3b50>
        # print("args.source_path: ", args.source_path)  # args:  <flowmap.arguments.ModelParams object at 0x7f8b1c0b3b50>

        self.model_path = args.model_path #将传入的 args 对象中的 model_path 属性赋值给 self.model_path，表示模型的路径。
        self.loaded_iter = None #初始化 self.loaded_iter 为 None，用于存储已加载的模型的迭代次数。
        self.gaussians = gaussians #将传入的高斯模型对象赋值给 self.gaussians 属性。

        if load_iteration: #可选参数，默认为 None。如果提供了值，它将被用作已加载模型的迭代次数。
            if load_iteration == -1: #如果没有提供 load_iteration，则将点云数据和相机信息保存到文件中。
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter)) #输出加载模型的迭代次数的信息。

        self.train_cameras = {}
        self.test_cameras = {}

        # 根据场景的类型（Colmap 或 Blender）加载相应的场景信息，存储在 scene_info 变量中。 ##### 根据colmap的输出文件读取场景信息：(训练和测试)内参、外参、点云、points3D.ply的路径、场景归一化参数
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # print("22222222222222")
            # print("args.source_path: ", args.source_path)     # /data/hkk/3dgs/gaussian-splatting/data
            # scene_info_start = time()                                     # args.image=image
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)   # sparse/cameras.bin  images.bin points3D.bin中读取文件
            # print("scene_info load time: ", time() - scene_info_start)     # 0.022114276885986328
            # print("scene_info:", scene_info)  # 根据colmap文件读取场景信息：点云、颜色、场景归一化参数、T、FovY、FovX、image参数(mode、size、image_path、image_name)、相机参数信息、
            # nerf_normalization、radius半径、ply_path(points3D.ply路径)
            # SceneInfo(point_cloud=BasicPointCloud(points=array([[-1.8414242, -1.9515846, 18.6913   ],  colors=array([[0.29803922, 0.54117647, 0.81568627]
            # normals=array([[0., 0., 0.],  train_cameras=[CameraInfo(uid=1, R=array([[ 0.71782023,  0.30135701, -0.62762893], ...
            # T=array([ 3.63733704,  1.83276296, -2.14316206]), FovY=1.2328371476448132, FovX=0.7321325820912802,
            # image=<PIL.PngImagePlugin.PngImageFile image mode=RGB size=711x1261 at 0x7FEE48BB28D0>, image_path='/data/hkk/3dgs/gaussian-splatting/data/images/1.png', image_name='1', width=711, height=1261)
            # CameraInfo(uid=1, R=array([[ 0.71824486,  0.30047758, -0.62756477],        [-0.29629856,  0.94816277,  0.1148674 ]
            # test_cameras=[], nerf_normalization={'translate': array([-0.18262084, -0.08079811,  0.01295633], dtype=float32), 'radius': 7.078606939315796}, ply_path='/data/hkk/3dgs/gaussian-splatting/data/sparse/0/points3D.ply') 
        
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")     # 也没有打印
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            # print("33333333333333333") 没有打印
            assert False, "Could not recognize scene type!"

        # 保存点云数据和相机信息：      # 保存为output/input.ply    output/cameras.json 
        if not os.path.exists(os.path.join("/data2/hkk/3dgs/flowmap/outputs/local/colmap", "output")):
            os.makedirs(os.path.join(self.model_path, "output"))
        # print("scene_info.ply_path: ", scene_info.ply_path)  # /data/hkk/3dgs/gaussian-splatting/data/sparse/0/points3D.ply
        # print("self.model_path ", self.model_path)  # None
        with open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            None
        if not self.loaded_iter:         
            # save_ply_start = time()
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                # print("successfully enter the with open")
                dest_file.write(src_file.read())
            # print("save_ply time: ", time() - save_ply_start)   #  0.017671585083007812
            json_cams = []
            camlist = []
            # camer_start = time()
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            # print("scene_info.train_cameras: ", scene_info.train_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                # print("3333333333311111111: ", )
                json_cams.append(camera_to_JSON(id, cam))
                # print("333333333337777777777: ", )
            # print("read camera time: ", time() - camer_start)   # 0.0002918243408203125
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                # print("3333333333322222222222: ", )
                json.dump(json_cams, file)

        shuffle_start = time()
        if shuffle:     #可选参数，默认为 True。如果设置为 True，则会对场景中的训练和测试相机进行随机排序。
            # print("3333333333333333333: ")
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        print("shuffle time: ", time() - shuffle_start)   #     
        # print("333333334444444444: ")
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 加载训练和测试相机：
        load_cameras_start = time()
        for resolution_scale in resolution_scales: #可选参数，默认为 [1.0]。一个浮点数列表，用于指定训练和测试相机的分辨率缩放因子。
            # print("Loading Training Cameras")
            # print("4444444444444444")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)  # 1.0
            # print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        print("load_cameras time: ", time() - load_cameras_start)   #   3.6159451007843018     ------这里浪费时间了！！！
        
        if self.loaded_iter:             # 如果已加载模型，则调用 load_ply 方法加载点云数据。
            # print("running load_ply--------------")
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:                              # 否则，调用 create_from_pcd 方法根据场景信息中的点云数据创建高斯模型。
            # print("running create_from_pcd--------------")   #  bingo  运行这个代码
            # create_from_pcd_start = time()
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            # print("create_from_pcd time: ", time() - create_from_pcd_start)          # 0.06915497779846191
            

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]