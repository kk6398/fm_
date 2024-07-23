import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from ..scene.gaussian_model import GaussianModel
from ..utils.sh_utils import eval_sh
from ..scene.cameras import Camera



# 这段代码是一个用于渲染场景的函数，主要是通过将高斯分布的点投影到2D屏幕上来生成渲染图像。
def render(viewpoint_camera: Camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    """
    viewpoint_camera:       # 是scene.py文件读取的sparse/cameras.bin  images.bin points3D.bin中获取相机参数
        full_proj_transform:
        image_height:1261
        image_name:
        image_width:
        original_image:
        projection_matrix:
        scale: 1.0
        training:True
        trans:array([0., 0., 0.])
        uid:31
        world_view_transform:
        zfar:100.0
        znear:0.01    
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means  # 
    """# 创建一个与输入点云（高斯模型）大小相同的零张量，用于记录屏幕空间中的点的位置。这个张量将用于计算对于屏幕空间坐标的梯度。"""
    # pc.get_xyz： 从GaussianModel，gaussian_model.py中create_from_pcd输出的xyz值
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # print("pipe: ", pipe)   # <arguments.GroupParams object at 0x7fe065fc76d0> [19/06 02:56:58]
    # Set up rasterization configuration
    # 计算视场的 tan 值，这将用于设置光栅化配置。
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # print("tanfovx: ", tanfovx)      # 0.3833445449846394
    # print("tanfovy: ", tanfovy)      # 0.7085160148877734
    
    # 设置光栅化的配置，包括图像的大小、视场的 tan 值、背景颜色、视图矩阵、投影矩阵、球面谐波、相机中心等。
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,                  # self.world_view_transform.inverse()[3, :3]
        prefiltered=False,
        debug=False
        # debug=pipe.debug
    )
    # print("raster_settings:", raster_settings)
    # image_height=1261, image_width=711, tanfovx=0.3833445449846394, tanfovy=0.7085160148877734, bg=tensor([0., 0., 0.], device='cuda:0'), scale_modifier=1.0,
    # viewmatrix=tensor([[ 0.7182,  0.3005, -0.6276,  0.0000], ..., [ 3.6417,  1.8329, -2.1219,  1.0000]], device='cuda:0')
    # projmatrix=tensor([[ 1.8736,  0.4241, -0.6276, -0.6276], ..., [ 9.4998,  2.5870, -2.1321, -2.1219]], device='cuda:0')
    # sh_degree=0, campos=tensor([-4.4980, -0.4152, -0.8483], device='cuda:0'), prefiltered=False, debug=False)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)    # 创建一个高斯光栅化器对象，用于将高斯分布投影到屏幕上。

    # 获取高斯分布的三维坐标、屏幕空间坐标和透明度。
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 如果提供了预先计算的3D协方差矩阵，则使用它。否则，它将由光栅化器根据尺度和旋转进行计算。
    scales = None
    rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:   # false
    #     # print("666666666666")       # 不读取
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)  # 获取预计算的三维协方差矩阵。
    # else:       # 获取缩放和旋转信息。（对应的就是3D高斯的协方差矩阵了）
    #     # print("7777777777777")    
    #     scales = pc.get_scaling
    #     rotations = pc.get_rotation

    scales = pc.get_scaling
    rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 如果提供了预先计算的颜色，则使用它们。否则，如果希望在Python中从球谐函数中预计算颜色，请执行此操作。如果没有，则颜色将通过光栅化器进行从球谐函数到RGB的转换。
    shs = None
    colors_precomp = None
    shs = pc.get_features
    # if override_color is None:
    #     if pipe.convert_SHs_python:
    #         shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)  # 将SH特征的形状调整为（batch_size * num_points，3，(max_sh_degree+1)**2）。
    #         dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))  # 计算相机中心到每个点的方向向量，并归一化。
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)                     # 计算相机中心到每个点的方向向量，并归一化。
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)              # 使用SH特征将方向向量转换为RGB颜色。
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)                             # 将RGB颜色的范围限制在0到1之间。
    #     else:
    #         shs = pc.get_features
    # else:
    #     colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # 调用光栅化器，将高斯分布投影到屏幕上，获得渲染图像和每个高斯分布在屏幕上的半径。
    # print("means3D:", means3D)   # [[-1.8414, -1.9516, 18.6913], ..., [13.7768,  5.6580, 15.5191]]
    # print("means2D:", means2D)   # [[0., 0., 0.], ..., [0., 0., 0.]]
    # print("opacity:", opacity)   # [[0.1000]]     # 随着训练优化，会发生变化
    # print("shs:", shs)           # [[-0.7159,  0.1460,  1.1191], [ 0.0000,  0.0000,  0.0000], [ 0.0000,  0.0000,  0.0000], ...
    # print("colors_precomp:", colors_precomp)    #  None
    # print("scales:", scales)                    # tensor([[0.1693, 0.1693, 0.1693], ...,  
    # print("rotations:", rotations)              # tensor([[1., 0., 0., 0.]
    # print("cov3D_precomp:", cov3D_precomp)      # None
    rendered_image, radii = rasterizer(
        means3D = means3D,                  # 从gaussian_model的create_from_pcd&pc.get_xy函数提取
        means2D = means2D,                  # pc.get_xy创建一个与输入点云（高斯模型）大小相同的零张量，用于记录屏幕空间中的点的位置。这个张量将用于计算对于屏幕空间坐标的梯度
        shs = shs,                          # pc.get_features
        colors_precomp = colors_precomp,
        opacities = opacity,                # pc.get_opacity
        scales = scales,                    # pc.get_scaling
        rotations = rotations,              # pc.get_rotation
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # 返回一个字典，包含渲染的图像、屏幕空间坐标、可见性过滤器（根据半径判断是否可见）以及每个高斯分布在屏幕上的半径。
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}








# from dataclasses import dataclass
# #
# import torch
# from einops import rearrange
# from jaxtyping import Float
# from torch import Tensor, nn

# from ..dataset.types import Batch
# from ..flow.flow_predictor import Flows
# from .backbone import BackboneCfg, get_backbone
# from .extrinsics import ExtrinsicsCfg, get_extrinsics
# from .intrinsics import IntrinsicsCfg, get_intrinsics
# from .projection import sample_image_grid, unproject




# @dataclass
# class ModelCfg:
#     backbone: BackboneCfg
#     intrinsics: IntrinsicsCfg
#     extrinsics: ExtrinsicsCfg
#     use_correspondence_weights: bool


# @dataclass
# class ModelOutput:    ##########################################
#     depths: Float[Tensor, "batch frame height width"]
#     surfaces: Float[Tensor, "batch frame height width xyz=3"]
#     intrinsics: Float[Tensor, "batch frame 3 3"]
#     extrinsics: Float[Tensor, "batch frame 4 4"]
#     backward_correspondence_weights: Float[Tensor, "batch frame-1 height width"]
    
#     # color: Float[Tensor, "batch frame 3 height width"]
#     # depth: Float[Tensor, "batch frame height width"] | None

# @dataclass
# class ModelExports:
#     extrinsics: Float[Tensor, "batch frame 4 4"]
#     intrinsics: Float[Tensor, "batch frame 3 3"]
#     colors: Float[Tensor, "batch frame 3 height width"]
#     depths: Float[Tensor, "batch frame height width"]





# class Model(nn.Module):
#     def __init__(
#         self,
#         cfg: ModelCfg,
#         num_frames: int | None = None,
#         image_shape: tuple[int, int] | None = None,
#     ) -> None:
#         super().__init__()
#         self.cfg = cfg
#         self.backbone = get_backbone(cfg.backbone, num_frames, image_shape)  # 输出深度和对应权重 backbone = BackboneOverfit(backbone_cfg, num_frames, image_shape) 
#         self.intrinsics = get_intrinsics(cfg.intrinsics)                     # 输出内参
#         self.extrinsics = get_extrinsics(cfg.extrinsics, num_frames)         # 输出外参

#     def forward(
#         self,
#         batch: Batch,
#         flows: Flows,
#         global_step: int,
#     ) -> ModelOutput:
#         device = batch.videos.device
#         _, _, _, h, w = batch.videos.shape

#         # 把前面flowmap计算参数的部分当作是encoder
#         # Run the backbone, which provides depths and correspondence weights.
#         backbone_out = self.backbone.forward(batch, flows)

#         # Allow the correspondence weights to be ignored as an ablation.
#         if not self.cfg.use_correspondence_weights:
#             backbone_out.weights = torch.ones_like(backbone_out.weights)

#         # Compute the intrinsics.
#         intrinsics = self.intrinsics.forward(batch, flows, backbone_out, global_step)

#         # Use the intrinsics to calculate camera-space surfaces (point clouds).
#         xy, _ = sample_image_grid((h, w), device=device)
#         surfaces = unproject(
#             xy,
#             backbone_out.depths,
#             rearrange(intrinsics, "b f i j -> b f () () i j"),
#         )

#         # Finally, compute the extrinsics.
#         extrinsics = self.extrinsics.forward(batch, flows, backbone_out, surfaces)

#         # In addition, render the color image and depth map via 3DGS according to the extrinsics, intrinsics and 3D points
#         # 渲染rgb图像和深度图像作为decoder
#         # gaussians参数在mvsplat中是通过encoder输出的高斯函数的属性：均值、协方差、harmonics、opacities
#         # 这里需要类似的做法：将folwmap输出的参数转换为一个包含均值、协方差、harmonics、opacities的Gaussian类
        
#         output = self.decoder.forward(                                               # 输出颜色和深度
#             gaussians,
#             batch["target"]["extrinsics"],
#             batch["target"]["intrinsics"],
#             batch["target"]["near"],
#             batch["target"]["far"],
#             (h, w),
#             depth_mode=self.train_cfg.depth_mode,
#         )
        
#         return ModelOutput(
#             backbone_out.depths,
#             surfaces,
#             intrinsics,
#             extrinsics,
#             backbone_out.weights,
#         )

#     @torch.no_grad()
#     def export(
#         self,
#         batch: Batch,
#         flows: Flows,
#         global_step: int,
#     ) -> ModelExports:
#         # For now, only implement exporting with a batch size of 1.
#         b, _, _, _, _ = batch.videos.shape
#         assert b == 1

#         output = self.forward(batch, flows, global_step)

#         return ModelExports(
#             output.extrinsics,
#             output.intrinsics,
#             batch.videos,
#             output.depths,
#         )
