from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_opacity import LossOpacity, LossOpacityCfgWrapper
from .loss_depth_gt import LossDepthGT, LossDepthGTCfgWrapper
from .loss_lod import LossLOD, LossLODCfgWrapper
from .loss_depth_consis import LossDepthConsis, LossDepthConsisCfgWrapper
from .loss_normal_consis import LossNormalConsis, LossNormalConsisCfgWrapper
#from .loss_chamfer_distance import LossChamferDistance, LossChamferDistanceCfgWrapper
from .loss_style_content import LossStyleContent, LossStyleContentCfgWrapper
from .loss_style import LossStyle, LossStyleCfgWrapper
from .loss_content import LossContent, LossContentCfgWrapper
from .loss_total_variance import LossTotalVariance, LossTotalVarianceCfgWrapper
from .loss_adain_style import LossAdaINStyle, LossAdaINStyleCfgWrapper
from .loss_clip import LossCLIP, LossCLIPCfgWrapper
from .loss_dinov3_content import LossDINOV3Content, LossDINOV3ContentCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossOpacityCfgWrapper: LossOpacity,
    LossDepthGTCfgWrapper: LossDepthGT,
    LossLODCfgWrapper: LossLOD,
    LossDepthConsisCfgWrapper: LossDepthConsis,
    LossNormalConsisCfgWrapper: LossNormalConsis,
    #LossChamferDistanceCfgWrapper: LossChamferDistance,
    LossStyleContentCfgWrapper: LossStyleContent,
    LossStyleCfgWrapper: LossStyle,
    LossAdaINStyleCfgWrapper: LossAdaINStyle,
    LossContentCfgWrapper: LossContent,
    LossTotalVarianceCfgWrapper: LossTotalVariance,
    LossCLIPCfgWrapper: LossCLIP,
    LossDINOV3ContentCfgWrapper: LossDINOV3Content,
}

LossCfgWrapper = (LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper | LossOpacityCfgWrapper |
                   LossDepthGTCfgWrapper | LossLODCfgWrapper | LossDepthConsisCfgWrapper | LossNormalConsisCfgWrapper |
                   LossStyleContentCfgWrapper | LossStyleCfgWrapper | LossContentCfgWrapper | LossDINOV3ContentCfgWrapper |
                   LossTotalVarianceCfgWrapper | LossAdaINStyleCfgWrapper | LossCLIPCfgWrapper)


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
