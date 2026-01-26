from typing import Optional, Union

from ..encoder import Encoder
from ..encoder.visualization.encoder_visualizer import EncoderVisualizer
from ..encoder.stylos import EncoderStylos, EncoderStylosCfg
from ..decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
from torch import nn
from .stylos import Stylos

MODELS = {
    "stylos": Stylos,
}

EncoderCfg = Union[EncoderStylosCfg]
DecoderCfg = DecoderSplattingCUDACfg


# hard code for now
def get_model(encoder_cfg: EncoderCfg, decoder_cfg: DecoderCfg) -> nn.Module:
    model = MODELS['stylos'](encoder_cfg, decoder_cfg)
    return model
