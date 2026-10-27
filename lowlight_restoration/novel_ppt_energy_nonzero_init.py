"""Initialization-only PPT energy repair; the frozen V1 implementation is untouched."""
from __future__ import annotations

from torch import nn

from .novel_ppt_energy import NovelPPTEnergy
from .vlm_energy_restoration import ConditionedNAFBlock

INTERNAL_RESIDUAL_INIT = 0.1


class NonzeroInitNovelPPTEnergy(NovelPPTEnergy):
    """Keep zero potential but open internal nonlinear feature paths.

    The constant is frozen for a mechanism test, not selected using development
    quality. No new parameters, forward computation, losses or RNG draws.
    Initial descent is still exactly identity because the last head stays zero.
    This does NOT make VLM conditioning active at initialization: the original
    modulation weights remain zero and must learn, unchanged from the control.
    """

    def __init__(self, width=24, condition_dim=96, residual_scale=.02,
                 step_size=.5, bound=4.):
        super().__init__(width, condition_dim, residual_scale, step_size, bound)
        for block in self.field.modules():
            if isinstance(block, ConditionedNAFBlock):
                nn.init.constant_(block.beta, INTERNAL_RESIDUAL_INIT)
                nn.init.constant_(block.gamma, INTERNAL_RESIDUAL_INIT)
        if bool(self.field.head.weight.detach().count_nonzero()):
            raise RuntimeError("initial output potential must remain zero")
