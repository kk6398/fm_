from dataclasses import dataclass



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
    