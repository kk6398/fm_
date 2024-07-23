import json
import shutil
from pathlib import Path
from time import time
import os
import hydra
import torch
import wandb
from jaxtyping import install_import_hook
from lightning import Trainer
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig
from torch.utils.data import default_collate
from argparse import ArgumentParser, Namespace

# from .arguments import ModelParams, PipelineParams, OptimizationParams
# import sys
# from .utils.general_utils import safe_state


# Configure beartype and jaxtyping.

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

with install_import_hook(
        ("flowmap",),
        ("beartype", "beartype"),
):
    from flowmap.config.common import get_typed_root_config
    from flowmap.config.overfit import OverfitCfg
    from flowmap.dataset import get_dataset
    from flowmap.dataset.data_module_overfit import DataModuleOverfit
    from flowmap.dataset.types import Batch
    from flowmap.export.colmap import export_to_colmap
    from flowmap.flow import compute_flows
    from flowmap.loss import get_losses
    from flowmap.misc.common_training_setup import run_common_training_setup
    from flowmap.misc.cropping import (
        crop_and_resize_batch_for_flow,
        crop_and_resize_batch_for_model,
    )
    from flowmap.model.model import Model
    from flowmap.model.model_wrapper_overfit import ModelWrapperOverfit
    from flowmap.tracking import compute_tracks
    from flowmap.visualization import get_visualizers

    # from .config.common import get_typed_root_config
    # from .config.overfit import OverfitCfg
    # from .dataset import get_dataset
    # from .dataset.data_module_overfit import DataModuleOverfit
    # from .dataset.types import Batch
    # from .export.colmap import export_to_colmap
    # from .flow import compute_flows
    # from .loss import get_losses
    # from .misc.common_training_setup import run_common_training_setup
    # from .misc.cropping import (
    #     crop_and_resize_batch_for_flow,
    #     crop_and_resize_batch_for_model,
    # )
    # from .model.model import Model
    # from .model.model_wrapper_overfit import ModelWrapperOverfit
    # from .tracking import compute_tracks
    # from .visualization import get_visualizers
from flowmap.scene import Scene, GaussianModel


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="overfit",
)
# def overfit(lp, op, pp, cfg_dict: DictConfig) -> None:
def overfit(cfg_dict: DictConfig) -> None:
    start_time = time()
    cfg = get_typed_root_config(cfg_dict, OverfitCfg)  # 获取cfg配置 参数
    print("cfg : ", cfg)
    print("cfg.dataset.root: ", cfg.dataset[0].root)  # /data2/hkk/datasets/flowmap/llff_fern
    # wandb=WandbCfg(project='flowmap', mode='disabled', name='placeholder', group=None, tags=None)
    # checkpoint=CheckpointCfg(every_n_train_steps=2000, load='checkpoints/initialization_finetuned.ckpt'),
    # trainer=TrainerCfg(val_check_interval=50, max_steps=2000),
    # flow=FlowPredictorRaftCfg(name='raft', num_flow_updates=32, max_batch_size=8, show_progress_bar=True),
    # dataset=[DatasetImagesCfg(image_shape=None, scene=None, name='images', root=PosixPath('/data2/hkk/datasets/flowmap/llff_fern'))],
    # frame_sampler=FrameSamplerOverfitCfg(name='overfit', start=None, num_frames=None, step=None),
    # model=ModelCfg(backbone=BackboneMidasCfg(name='midas', pretrained=True, weight_sensitivity=None, mapping='original', model='MiDaS_small'),
    # intrinsics=IntrinsicsSoftminCfg(name='softmin', num_procrustes_points=8192, min_focal_length=0.5, max_focal_length=2.0, num_candidates=60, regression=RegressionCfg(after_step=1000, window=100)),
    # extrinsics=ExtrinsicsProcrustesCfg(name='procrustes', num_points=1000, randomize_points=False),
    # use_correspondence_weights=True),
    # loss=[LossFlowCfg(enable_after=0, weight=1000.0, name='flow', mapping=MappingHuberCfg(name='huber', delta=0.01)),
    #       LossTrackingCfg(enable_after=50, weight=100.0, name='tracking', mapping=MappingHuberCfg(name='huber', delta=0.01))],
    # visualizer=[VisualizerSummaryCfg(name='summary', num_vis_frames=8),
    # VisualizerTrajectoryCfg(name='trajectory', generate_plot=True, ate_save_path=None)],
    # cropping=CroppingCfg(image_shape=43200, flow_scale_multiplier=4, patch_size=32),
    # tracking=TrackPredictorCoTrackerCfg(name='cotracker', grid_size=35, similarity_threshold=0.2),
    # track_precomputation=TrackPrecomputationCfg(cache_path=None, interval=5, radius=20),
    # model_wrapper=ModelWrapperOverfitCfg(lr=3e-05, patch_size=32), local_save_root=None)

    # 初始化guassians模型
    gaussians = GaussianModel(cfg.trainer.sh_degree)
    bg_color = [1, 1, 1] if cfg.trainer.white_background else [0, 0, 0]  # 设置背景颜色，根据数据集是否有白色背景来选择。     # white_background：False
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 将背景颜色转化为 PyTorch Tensor，并移到 GPU 上

    callbacks, logger, checkpoint_path, output_dir = run_common_training_setup(
        # 获取回调函数(监控学习率的变化)、日志记录器、checkpoints路径、输出目录
        cfg, cfg_dict
    )
    device = torch.device("cuda:0")
    colmap_path = output_dir / "colmap"

    # Load the full-resolution batch.
    dataset = get_dataset(cfg.dataset, "train", cfg.frame_sampler)  # 获取数据集
    # print("dataset: ", dataset)                                  # dataset:  <flowmap.dataset.dataset_merged.DatasetMerged object at 0x7fb2be2e2710>
    batch = next(iter(dataset))  # 获取数据集的一个batch
    # 'frame_paths': 20张图像的路径 [PosixPath('/data2/hkk/datasets/flowmap/llff_fern/IMG_4026.JPG'),
    frame_paths = batch.pop("frame_paths", None)  # 获取batch中的frame_paths, 即每个视频帧的路径
    # print("----------------------------------")
    # print("frame_paths: ", frame_paths)                      # frame_paths:  [PosixPath('/data2/hkk/datasets/flowmap/llff_fern/IMG_4026.JPG'),
    if frame_paths is not None:
        frame_paths = [Path(path) for path in
                       frame_paths]  # 将frame_paths转换为Path对象    使用 Path 对象可以更方便地进行文件路径的操作，例如获取文件名、父目录、扩展名等。
    batch = Batch(**default_collate([batch]))  # 将batch中的数据进行拼接

    # Compute optical flow and tracks.                      # batch_for_model: 裁剪后的batch、图像内参、videos
    batch_for_model, pre_crop = crop_and_resize_batch_for_model(batch,       # batch_for_model.videos  torch.Size([1, 20, 3, 160, 224])    # pre_crop:  (180, 240)
                                                                cfg.cropping)  # 对batch进行裁剪和缩放  ###    pre_crop:裁剪后的H W

    batch_for_flow = crop_and_resize_batch_for_flow(batch, cfg.cropping)  # 对batch进行裁剪和缩放， 针对光流  #  batch_for_flow.videos  torch.Size([1, 20, 3, 640, 896])

    _, f, _, h, w = batch_for_model.videos.shape  # 获取batch中的视频的帧数、高、宽
    flows = compute_flows(batch_for_flow, (h, w), device, cfg.flow)  # 计算光流,  输入是裁剪之后的batch, 图像的高和宽， 设备， cfg.flow

    # Only compute tracks if the tracking loss is enabled.
    if any([loss.name == "tracking" for loss in cfg.loss]):  # 如果损失函数中有tracking
        tracks = compute_tracks(  # 计算tracks    # CoTracker V1
            batch_for_flow, device, cfg.tracking, cfg.track_precomputation
        )
    else:
        tracks = None

    # Set up the model.
    optimization_start_time = time()  #
    # model = Model(cfg.model, num_frames=f, image_shape=(h, w))
    model = Model(cfg.model, num_frames=f, image_shape=(h, w))  # 模型初始化
    losses = get_losses(cfg.loss)  # loss初始化    # 对flow、tracking损失进行带有权重初始化的设置
    visualizers = get_visualizers(cfg.visualizer)  # 可视化
    model_wrapper = ModelWrapperOverfit(  # 模型包装器
        cfg.model_wrapper,  # ModelWrapperOverfitCfg(lr=3e-05, patch_size=32), local_save_root=None)
        model,  # 初始化的模型
        batch_for_model,  # crop后的batch，包括videos中每帧的具体数据, scenes, extrinsics、intrinsics
        flows,  # 计算出的光流
        tracks,  # 计算出的 tracks: CoTracker V1     facebookresearch_co-tracker_v1.0
        losses,  # 损失函数
        visualizers,

        gaussians,       # gaussian 模型
        background,      # 背景颜色
        frame_paths,     # 所有图像的路径
        pre_crop,        # 裁剪后的H W
        batch.videos,    # original rgb图像          batch.videos.shape   [1, 20, 3, 3024, 4032]
        colmap_path,     # colmap路径    未用到
        cfg.dataset[0].root,    # 读取图像的上级路径      # /data2/hkk/datasets/flowmap/llff_fern
        cfg.trainer             # 包含了optimizer的参数
    )  # 输出: 整个model模型

    # Only load the model's saved state (so that optimization restarts).
    if checkpoint_path is not None:  # 如果有checkpoint路径
        checkpoint = torch.load(checkpoint_path)
        model_wrapper.load_state_dict(checkpoint["state_dict"], strict=False)

    trainer = Trainer(  # PyTorch Lightning 训练器初始化 #
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(  # 策略：是否使用ddp分布式训练
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,  # 50
        max_steps=cfg.trainer.max_steps,  # 2000，        30000
        log_every_n_steps=1,
    )
    trainer.fit(  # 训练
        model_wrapper,
        datamodule=DataModuleOverfit(),
    )

    # Export the result.
    print("Exporting results.")



    ### 后续的生成colmap文件，，未用到
    model_wrapper.to(device)
    exports = model_wrapper.export(device)
    colmap_path = output_dir / "colmap"
    export_to_colmap(
        exports,
        frame_paths,
        pre_crop,
        batch.videos,
        colmap_path,
    )
    shutil.make_archive(colmap_path, "zip", output_dir, "colmap")  # 压缩文件

    if cfg.local_save_root is not None:  # 是None     #  如果有本地保存路径
        # Save the COLMAP-style output.
        cfg.local_save_root.mkdir(exist_ok=True, parents=True)  # 创建本地保存路径
        shutil.copytree(colmap_path, cfg.local_save_root, dirs_exist_ok=True)  # 复制文件

        # Save the runtime. For a fair comparison with COLMAP, we record how long it
        # takes until the COLMAP-style output has been saved.
        times = {  # 记录时间
            "runtime": time() - start_time,
            "optimization_runtime": time() - optimization_start_time,
        }
        with (cfg.local_save_root / "runtime.json").open("w") as f:  # 保存时间
            json.dump(times, f)

            # Save the exports (poses, intrinsics, depth maps + corresponding colors).
        torch.save(exports, cfg.local_save_root / "exports.pt")  # 保存导出结果

        # Save a checkpoint.
        trainer.save_checkpoint(cfg.local_save_root / "final.ckpt")  # 保存checkpoint

    if cfg.wandb.mode != "disabled":  # 如果wandb不是禁用状态
        artifact = wandb.Artifact(f"colmap_{wandb.run.id}", type="colmap")  # 创建wandb artifact
        artifact.add_file(f"{colmap_path}.zip", name="colmap.zip")  # 添加文件
        wandb.log_artifact(artifact)  # 记录artifact
        artifact.wait()  # 等待artifact


if __name__ == "__main__":  # 主函数
    overfit()

