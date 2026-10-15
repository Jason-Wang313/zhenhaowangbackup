from __future__ import annotations

import pytest
import torch

from lowlight_restoration.deblur_expert import DeblurPixelExpert, deblur_expert_parameter_count


def test_deblur_expert_starts_as_exact_identity() -> None:
    model = DeblurPixelExpert(width=8, encoder_blocks=(1, 1), middle_blocks=1, decoder_blocks=(1, 1))
    image = torch.rand(2, 3, 32, 36)
    assert torch.equal(model(image), image)
    assert deblur_expert_parameter_count(model) > 0


def test_deblur_expert_rejects_unaligned_geometry() -> None:
    model = DeblurPixelExpert(width=8, encoder_blocks=(1, 1), middle_blocks=1, decoder_blocks=(1, 1))
    with pytest.raises(ValueError, match="divide by four"):
        model(torch.rand(1, 3, 31, 32))
