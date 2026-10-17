"""Restoration-native proposal residual basis for the post-V2 causal test."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .task_native_restormer import Restormer


class ProposalResidualBasisRestormer(Restormer):
    """Two independently supervised corrections around one frozen proposal.

    The nine input channels are observation, proposal, and proposal-observation.
    The two RGB outputs form a fidelity-oriented and perceptual-oriented basis.
    """

    def __init__(self, *, residual_bound: float = 0.25, initial_output_scale: float = 0.1) -> None:
        super().__init__(
            input_channels=9,
            output_channels=6,
            layer_norm_type="BiasFree",
        )
        self.residual_bound = float(residual_bound)
        self.initial_output_scale = float(initial_output_scale)

    def initialize_from_proposal(self, state: Mapping[str, Tensor]) -> None:
        own = self.state_dict()
        special = {"patch_embed.proj.weight", "output.weight"}
        if set(state) - special != set(own) - special:
            raise RuntimeError("proposal residual basis backbone contract changed")
        copied = {}
        for key, value in own.items():
            if key not in special:
                source = state[key]
                if source.shape != value.shape:
                    raise RuntimeError(f"proposal residual basis shape changed: {key}")
                copied[key] = source.detach().clone()
        patch = torch.zeros_like(own["patch_embed.proj.weight"])
        source_patch = state["patch_embed.proj.weight"]
        patch[:, 3:6] = source_patch
        output = (
            torch.cat((state["output.weight"], state["output.weight"]), dim=0)
            * self.initial_output_scale
        )
        copied["patch_embed.proj.weight"] = patch
        copied["output.weight"] = output
        self.load_state_dict(copied, strict=True)

    def forward(self, observation: Tensor, proposal: Tensor) -> tuple[Tensor, Tensor]:
        if observation.shape != proposal.shape or observation.ndim != 4 or observation.shape[1] != 3:
            raise ValueError("observation and proposal must be matching BCHW RGB tensors")
        if observation.shape[-2] % 8 or observation.shape[-1] % 8:
            raise ValueError("proposal residual basis spatial dimensions must divide by 8")
        image = torch.cat((observation, proposal, proposal - observation), dim=1)
        encoder1 = self.encoder_level1(self.patch_embed(image))
        encoder2 = self.encoder_level2(self.down1_2(encoder1))
        encoder3 = self.encoder_level3(self.down2_3(encoder2))
        latent = self.latent(self.down3_4(encoder3))
        decoder3 = self.decoder_level3(
            self.reduce_chan_level3(torch.cat((self.up4_3(latent), encoder3), dim=1))
        )
        decoder2 = self.decoder_level2(
            self.reduce_chan_level2(torch.cat((self.up3_2(decoder3), encoder2), dim=1))
        )
        decoder1 = self.refinement(
            self.decoder_level1(torch.cat((self.up2_1(decoder2), encoder1), dim=1))
        )
        fidelity_residual, perceptual_residual = self.output(decoder1).chunk(2, dim=1)
        fidelity = (proposal + self.residual_bound * torch.tanh(fidelity_residual)).clamp(0, 1)
        perceptual = (proposal + self.residual_bound * torch.tanh(perceptual_residual)).clamp(0, 1)
        return fidelity, perceptual


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
