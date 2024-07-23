##
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from ..utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from ..utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from ..utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from ..utils.graphics_utils import BasicPointCloud
from ..utils.general_utils import strip_symmetric, build_scaling_rotation
import open3d as o3d
# from .cameras import Camera
import collections


# from open3d.utility import Quaternion

# Camera_viewpoints = collections.namedtuple("Camera_viewpoints", ["id", "model", "width", "height", "params"])

class GaussianModel:

    def setup_functions(self):  # 用于设置一些激活函数和变换函数
        # 构建协方差矩阵，该函数接受 scaling（尺度）、scaling_modifier（尺度修正因子）、rotation（旋转）作为参数
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm  # 最终返回对称的协方差矩阵。

        self.scaling_activation = torch.exp  # 将尺度激活函数设置为指数函数。
        self.scaling_inverse_activation = torch.log  # 将尺度逆激活函数设置为对数函数。

        self.covariance_activation = build_covariance_from_scaling_rotation  # 将协方差激活函数设置为上述定义的 build_covariance_from_scaling_rotation 函数。

        self.opacity_activation = torch.sigmoid  # 将不透明度激活函数设置为 sigmoid 函数。  [0,1]
        self.inverse_opacity_activation = inverse_sigmoid  # 将不透明度逆激活函数设置为一个名为 inverse_sigmoid 的函数

        self.rotation_activation = torch.nn.functional.normalize  # 用于归一化旋转矩阵。

    def __init__(self, sh_degree: int):
        """
        初始化3D高斯模型的参数。

        :param sh_degree: 球谐函数的最大次数，用于控制颜色表示的复杂度。
        """
        # 初始化球谐次数和最大球谐次数
        self.active_sh_degree = 0  # 当前激活的球谐次数，初始为0
        self.max_sh_degree = sh_degree  # 允许的最大球谐次数

        # 初始化3D高斯模型的各项参数
        self._xyz = torch.empty(0)  # 3D高斯的中心位置（均值）
        self._features_dc = torch.empty(0)  # 第一个球谐系数，用于表示基础颜色
        self._features_rest = torch.empty(0)  # 其余的球谐系数，用于表示颜色的细节和变化
        self._scaling = torch.empty(0)  # 3D高斯的尺度参数，控制高斯的宽度
        self._rotation = torch.empty(0)  # 3D高斯的旋转参数，用四元数表示
        self._opacity = torch.empty(0)  # 3D高斯的不透明度，控制可见性
        self.max_radii2D = torch.empty(0)  # 在2D投影中，每个高斯的最大半径
        self.xyz_gradient_accum = torch.empty(0)  # 用于累积3D高斯中心位置的梯度
        self.denom = torch.empty(0)  # 未明确用途的参数
        self.optimizer = None  # 优化器，用于调整上述参数以改进模型
        self.percent_dense = 0  # 设置在训练过程中，用于密集化处理的3D高斯点的比例
        self.spatial_lr_scale = 0

        # 调用setup_functions来初始化一些处理函数
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):  # 每 1000 次迭代，增加球谐函数的阶数。
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):  # 用于从给定的点云数据 pcd 创建对象的初始化状态。
        """
        从点云数据初始化模型参数。

        :param pcd: 点云数据，包含点的位置和颜色。
        :param spatial_lr_scale: 空间学习率缩放因子，影响位置参数的学习率。
        """
        # print("55555555555555555")
        # 将点云的位置和颜色数据从numpy数组转换为PyTorch张量，并传送到CUDA设备上
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()  # (P, 3) # 点云的位置数据 具体数据形式表示是什么？？？
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())  # (P, 3) # 点云的颜色数据 具体数据形式表示是什么？？？

        # 初始化存储球谐系数的张量，每个颜色通道有(max_sh_degree + 1) ** 2个球谐系数
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()  # (P, 3, 16)
        features[:, :3, 0] = fused_color  # 将RGB转换后的球谐系数C0项的系数存入
        features[:, 3:, 1:] = 0.0  # 其余球谐系数初始化为0

        # print("Number of points at initialisation : ", fused_point_cloud.shape[0])    # 打印初始点的数量    28478

        # 计算点云中每个点到其最近的k个点的平均距离的平方，用于确定高斯的尺度参数
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)  # (P,)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)  # (P, 3)

        # 初始化每个点的旋转参数为单位四元数（无旋转）
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")  # (P, 4)
        rots[:, 0] = 1  # 四元数的实部为1，表示无旋转

        # 初始化每个点的不透明度为0.1（通过inverse_sigmoid转换）
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 将以上计算的参数设置为模型的可训练参数
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))  # 位置
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))  # 球谐系数C0项
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))  # 其余球谐系数
        self._scaling = nn.Parameter(scales.requires_grad_(True))  # 尺度
        self._rotation = nn.Parameter(rots.requires_grad_(True))  # 旋转
        self._opacity = nn.Parameter(opacities.requires_grad_(True))  # 不透明度
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")  # 存储2D投影的最大半径，初始化为0

        # print("self._xyz", self._xyz.shape)                       # [28478, 3] # tensor([[-1.8414, -1.9516, 18.6913], ..., [13.7768,  5.6580, 15.5191]], device='cuda:0', requires_grad=True)
        # print("self._features_dc", self._features_dc.shape)     # [28478, 1, 3] # tensor([[[-0.7159,  0.1460,  1.1191]], ..., [[ 0.1043, -0.2016, -0.9523]]], device='cuda:0', requires_grad=True)
        # print("self._features_rest", self._features_rest.shape) # [28478, 15, 3]  # tensor([[[0., 0., 0.], ..., [0., 0., 0.]]], device='cuda:0', requires_grad=True)
        # print("self._scaling", self._scaling.shape)             # [28478, 3] # tensor([[-1.7758, -1.7758, -1.7758], ..., [-2.1550, -2.1550, -2.1550]], device='cuda:0', requires_grad=True)
        # print("self._rotation", self._rotation.shape)           # [28478, 4] # tensor([[1., 0., 0., 0.], ..., [1., 0., 0., 0.]], device='cuda:0', requires_grad=True)
        # print("self._opacity", self._opacity.shape)             # [28478, 1] # tensor([[-2.1972], ..., [-2.1972]], device='cuda:0', requires_grad=True)
        # print("self.max_radii2D", self.max_radii2D.shape)       # [28478] # tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0')

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")   # [560,1]
        # print("self.xyz_gradient_accum.shape00000", self.xyz_gradient_accum.shape)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")              # [560,1]

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):  # 更新学习率
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):  # 这个方法的目的是从PLY文件中加载各种数据，并将这些数据存储为类中的属性，以便后续的操作和训练。
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # print("update_filter:", update_filter)  # device='cuda:0'
        # print("viewspace_point_tensor.grad:", viewspace_point_tensor.grad)  # None
        # print("update_filter.shape:",
        #       update_filter.shape)  # device='cuda:0'    torch.Size([560])      # torch.Size([560])
        # print("viewspace_point_tensor.grad.shape:", viewspace_point_tensor.grad.shape)  # torch.Size([560, 3])
        # print("self.xyz_gradient_accum.shape:", self.xyz_gradient_accum.shape)
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    # def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False):
    # rgb [B, N, C, H, W]    torch.Size([1, 20, 3, 160, 224])
    # depth [C, N, H, W]     torch.Size([1, 20, 160, 224])
    # intrinsics [C, N, 3, 3]
    # extrinsics [C, N, 4, 4]
    def create_pcd_from_image_and_depth(self, rgb, depths, intrinsics, extrinsics):
        rgb = rgb.squeeze(0)       # [20, 3, 160, 224]
        depths = depths.squeeze(0)    # torch.Size([20, 160, 224])
        # print("rgb.shape", rgb.shape)         # torch.Size([20, 3, 160, 224])
        # print("depths.shape", depths.shape)   # torch.Size([20, 160, 224])
        point_clouds = []
        # final_pcd = []
        for i in range(rgb.shape[0]):
            # 提取第 i 个图像的RGB和深度图像
            rgb_image_split = rgb[i, :, :, :]   # [3, 160, 224]  ~ (0, 1)
            rgb_image_ab = torch.clamp(rgb_image_split, 0.0, 1.0)
            rgb_image = (rgb_image_ab * 255).byte().permute(1, 2, 0).contiguous().detach().cpu().numpy()  # [160,224,3] ~ (0, 255)

            depth_image = depths[i, :, :]      # [160, 224]  tensor  数值很小0~3
            depth_image = depth_image.contiguous().detach().cpu().numpy()     # array
            # print("rgb_image.shape", rgb_image.shape)         # (160, 224, 3)
            # print("depth_image.shape", depth_image.shape)     # (160, 224)

            # 将图像数据转换为Open3D的Image对象
            rgb_image_o3d = o3d.geometry.Image(rgb_image.astype(np.uint8))
            depth_image_o3d = o3d.geometry.Image(depth_image.astype(np.float32))

            # 创建RGBD图像
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_image_o3d,
                depth_image_o3d,
                depth_scale=1.0,
                depth_trunc=100.0,
                convert_rgb_to_intensity=False,
            )

            # print("intrinsic.shape", intrinsics.shape)        # torch.Size([1, 20, 3, 3])
            # print("extrinsic.shape", extrinsics.shape)        # torch.Size([1, 20, 4, 4])
            intrinsic_sq = intrinsics.squeeze(0)  # torch.Size([20, 3, 3])
            extrinsic_sq = extrinsics.squeeze(0)  # torch.Size([20, 4, 4])
            intrinsic = intrinsic_sq[i, :, :]    # [3, 3]
            extrinsic = extrinsic_sq[i, :, :]    # [4, 4]
            intrinsic_array = intrinsic.contiguous().detach().cpu().numpy()
            extrinsic_array = extrinsic.contiguous().detach().cpu().numpy()

            # # 根据输入 rgbd、相机内参和外参，创建点云
            pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    width=rgb.shape[-1],       # 224
                    height=rgb.shape[-2],      # 160
                    fx=intrinsic_array[0, 0],
                    fy=intrinsic_array[1, 1],
                    cx=intrinsic_array[0, 2],
                    cy=intrinsic_array[1, 2],
                ),
                extrinsic_array,
            )
            point_clouds.append(pcd_tmp)    # 这行代码是将0~20的点云组成一个字典的索引

        # final_pcd = np.concatenate(point_clouds)    # torch.cat    np.concatenate
        # 假设所有点云都使用相同的坐标系。如果点云在不同的坐标系中，你可能需要首先将它们转换到同一个坐标系中，然后再进行合并。

        final_pcd = o3d.geometry.PointCloud()    # 35840*20=716800
        for pcd in point_clouds:
            final_pcd += pcd

        # 对点云进行随机下采样。
        # pcd_tmp = pcd_tmp.random_down_sample(1.0 / 64)  # downsample_factor:64
        # 获取点云的坐标和颜色信息。
        new_xyz = np.asarray(final_pcd.points)   # [716800,3]
        new_rgb = np.asarray(final_pcd.colors)   # [716800,3]

        # 创建一个基本的点云对象。
        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )

        # 将点云的点坐标转换为 PyTorch 张量，并移到GPU上。
        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()   # [716800,3]
        # 将点云的颜色信息转换为 PyTorch 张量，然后再转成sh系数，并移到GPU上。
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda()) # [716800,3]
        # 初始化点云的特征表示。
        features = (                 # # [716800,3,16] # *size   [a, b, c]
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))  # max_sh_degree 3或0
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        # 根据点与相机之间的距离计算尺度。 #####################
        dist2 = (
            torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            # * point_size    # point_size: 0.01
        )
        # scales = torch.log(torch.sqrt(dist2))[..., None]
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)  # [716800,3]

        # 初始化旋转矩阵。
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")  # # [716800,4]# 大小为fused_point_cloud.shape[0]行、4列的零矩阵
        rots[:, 0] = 1
        # 初始化不透明度。
        opacities = inverse_sigmoid(      # # [716800,1]
            0.5            # original gs: 0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))  # 位置
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))  # 球谐系数C0项
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))  # 其余球谐系数
        self._scaling = nn.Parameter(scales.requires_grad_(True))  # 尺度
        self._rotation = nn.Parameter(rots.requires_grad_(True))  # 旋转
        self._opacity = nn.Parameter(opacities.requires_grad_(True))  # 不透明度
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
