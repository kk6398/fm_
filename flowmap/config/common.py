from dataclasses import dataclass
from typing import Literal, Type, TypeVar

from omegaconf import DictConfig


from ..dataset import DatasetCfg
from ..flow import FlowPredictorCfg
from ..frame_sampler import FrameSamplerCfg
from ..loss import LossCfg
from ..misc.cropping import CroppingCfg
from ..model.model import ModelCfg
from ..visualization import VisualizerCfg
from .tools import get_typed_config, separate_multiple_defaults


@dataclass
class WandbCfg:                           #  Python类的定义 WandbCfg   类中的每个属性都有一个类型注解
    project: str
    mode: Literal["online", "offline", "disabled"]
    name: str | None
    group: str | None
    tags: list[str] | None


@dataclass
class CheckpointCfg:
    every_n_train_steps: int
    load: str | None  # str instead of Path, since it could be wandb://...


@dataclass
class TrainerCfg:
    val_check_interval: int     # 验证 检查 间隔      
    max_steps: int              # 最大步数
    
    # 在此添加opt优化器的参数配置
    iterations: int
    position_lr_init: float
    position_lr_final: float
    position_lr_delay_mult: float
    position_lr_max_steps: int
    feature_lr: float
    opacity_lr: float
    scaling_lr: float
    rotation_lr: float
    percent_dense: float
    lambda_dssim: float
    densification_interval: int
    opacity_reset_interval: int
    densify_from_iter: int
    densify_until_iter: int
    densify_grad_threshold: float
    
    # 在此添加model模型的参数配置
    sh_degree: int                  # or  3?
    source_path: str
    model_path: str
    resolution: int
    white_background: bool
    data_device: str
    
    # 在此添加pipe的参数配置
    convert_SHs_python: bool
    compute_cov3D_python: bool
    


# wandb=WandbCfg(project='flowmap', mode='disabled', name='placeholder', group=None, tags=None) 
# checkpoint=CheckpointCfg(every_n_train_steps=2000, load='checkpoints/initialization_finetuned.ckpt'), # 
# trainer=TrainerCfg(val_check_interval=50, max_steps=2000), 
# flow=FlowPredictorRaftCfg(name='raft', num_flow_updates=32, max_batch_size=8, show_progress_bar=True), 
# dataset=[DatasetImagesCfg(image_shape=None, scene=None, name='images', root=PosixPath('/data2/hkk/datasets/flowmap/llff_fern'))], 
# frame_sampler=FrameSamplerOverfitCfg(name='overfit', start=None, num_frames=None, step=None), 
# model=ModelCfg(backbone=BackboneMidasCfg(name='midas', pretrained=True, weight_sensitivity=None, mapping='original', model='MiDaS_small'), 
   # intrinsics=IntrinsicsSoftminCfg(name='softmin', num_procrustes_points=8192, min_focal_length=0.5, max_focal_length=2.0, num_candidates=60, regression=RegressionCfg(after_step=1000, window=100)), 
   # extrinsics=ExtrinsicsProcrustesCfg(name='procrustes', num_points=1000, randomize_points=False), 
   # use_correspondence_weights=True), 
# loss=[LossFlowCfg(enable_after=0, weight=1000.0, name='flow', mapping=MappingHuberCfg(name='huber', delta=0.01)), LossTrackingCfg(enable_after=50, weight=100.0, name='tracking', mapping=MappingHuberCfg(name='huber', delta=0.01))], 
# visualizer=[VisualizerSummaryCfg(name='summary', num_vis_frames=8), 
# VisualizerTrajectoryCfg(name='trajectory', generate_plot=True, ate_save_path=None)], 
# cropping=CroppingCfg(image_shape=43200, flow_scale_multiplier=4, patch_size=32), 
# tracking=TrackPredictorCoTrackerCfg(name='cotracker', grid_size=35, similarity_threshold=0.2), 
# track_precomputation=TrackPrecomputationCfg(cache_path=None, interval=5, radius=20), 
# model_wrapper=ModelWrapperOverfitCfg(lr=3e-05, patch_size=32), local_save_root=None)


@dataclass
class CommonCfg:
    wandb: WandbCfg                 # wandb配置：项目project、模式mode、名称name、组group、标签tags  
    checkpoint: CheckpointCfg       # 检查点配置：每n个训练步骤保存一次检查点every_n_train_steps、加载检查点load
    trainer: TrainerCfg             # 训练器配置：验证检查间隔val_check_interval、最大步数max_steps
    flow: FlowPredictorCfg          # 光流预测器配置: name、num_flow_updates、 max_batch_size, show_progress_bar
    dataset: list[DatasetCfg]       
    # 根据不同的数据集，配置不同的参数 例如：DatasetCO3DCfg、DatasetCOLMAPCfg、DatasetImagesCfg、DatasetLLFFCfg、DatasetRE10kCfg ，但好像name都是‘images’
    # 例如：image_shape=None, scene=None, name='images', root=
    frame_sampler: FrameSamplerCfg  # 帧采样（光流全部帧都计算，新视角合成可能需要90帧）：frame_sampler=FrameSamplerOverfitCfg(name='overfit', start=None, num_frames=None, step=None)
    model: ModelCfg                 # 模型参数配置，包括model、内参的三种求解方式：GT、softmin、Regressed(前1000step softmin，后面用回归); 外参的两种求解方式：GT、Regressed；use_correspondence_weights(这个应该是MLP输出的权重)
    loss: list[LossCfg]             # 损失函数配置，包括flow、tracking，每个损失函数的权重为1000.0、100.0，采用的映射函数为huber
    visualizer: list[VisualizerCfg] # 可视化配置，包括summary、trajectory
    cropping: CroppingCfg           # 裁剪配置，包括image_shape=43200、flow_scale_multiplier=4、patch_size=32
    


T = TypeVar("T")


def get_typed_root_config(cfg_dict: DictConfig, cfg_type: Type[T]) -> T:
    return get_typed_config(
        cfg_type,
        cfg_dict,
        {
            list[DatasetCfg]: separate_multiple_defaults(DatasetCfg),
            list[LossCfg]: separate_multiple_defaults(LossCfg),
            list[VisualizerCfg]: separate_multiple_defaults(VisualizerCfg),
        },
    )
