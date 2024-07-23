from dataclasses import dataclass
from pathlib import Path

from ..model.model_wrapper_overfit import ModelWrapperOverfitCfg
from ..tracking import TrackPrecomputationCfg, TrackPredictorCfg
from .common import CommonCfg
#

@dataclass
class OverfitCfg(CommonCfg):         # 继承自 CommonCfg。这意味着 OverfitCfg 将包含 CommonCfg 中定义的所有属性和方法
    tracking: TrackPredictorCfg
    track_precomputation: TrackPrecomputationCfg
    model_wrapper: ModelWrapperOverfitCfg
    local_save_root: Path | None
    # opt_params: OptparamsCfg