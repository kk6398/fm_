from .backbone import Backbone
from .backbone_explicit_depth import BackboneExplicitDepth, BackboneExplicitDepthCfg
from .backbone_midas import BackboneMidas, BackboneMidasCfg
#
BACKBONES = {                                   # 从这个字典中选一个，但默认选择的是BackboneMidas
    "explicit_depth": BackboneExplicitDepth,             
    "midas": BackboneMidas, 
}
 
BackboneCfg = BackboneExplicitDepthCfg | BackboneMidasCfg     # 两个里面选一个


def get_backbone(
    cfg: BackboneCfg,
    num_frames: int | None,
    image_shape: tuple[int, int] | None,
) -> Backbone:
    return BACKBONES[cfg.name](cfg, num_frames, image_shape)
