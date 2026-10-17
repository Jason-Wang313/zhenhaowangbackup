import json
import unittest
from pathlib import Path

import torch

from lowlight_restoration.pairwise_safe_candidate_gate import (
    PairwiseSafeCandidateGate,
    compose_pairwise_selection,
    pairwise_targets,
    select_pairwise_candidates,
)
from smoke.train_foundir_pairwise_safe_candidate_gate_noise_v1_ddp import select_calibration_policy


ROOT = Path(__file__).resolve().parents[1]


class PairwiseSafeCandidateGateTest(unittest.TestCase):
    def test_initial_gate_returns_fidelity(self):
        model = PairwiseSafeCandidateGate(alternative_count=2, widths=(16,24,32,40,48), blocks=1)
        observation=torch.rand(1,3,64,64); proposal=torch.rand_like(observation); fidelity=torch.rand_like(observation)
        alternatives=torch.rand(1,2,3,64,64); outputs=model(observation,proposal,fidelity,alternatives,torch.rand(1,5))
        selected,accepted=select_pairwise_candidates(outputs,gain_probability=.5,maximum_harm_probability=.5)
        restored=compose_pairwise_selection(fidelity,alternatives,selected,accepted)
        self.assertEqual(outputs.shape,(1,2,3,2,2)); self.assertFalse(bool(accepted.any())); self.assertTrue(torch.equal(restored,fidelity))

    def test_targets_identify_gain_and_harm(self):
        target=torch.zeros(1,3,32,32); fidelity=torch.full_like(target,.2)
        alternatives=torch.stack((torch.full_like(target,.1),torch.full_like(target,.3)),dim=1)
        fidelity_perceptual=torch.full((1,32,32),.2); alternative_perceptual=torch.stack((torch.full_like(fidelity_perceptual,.1),torch.full_like(fidelity_perceptual,.4)),dim=1)
        mse,perceptual=pairwise_targets(fidelity,alternatives,target,fidelity_perceptual,alternative_perceptual)
        self.assertLess(float(mse[0,0,0,0]),0); self.assertGreater(float(mse[0,1,0,0]),0)
        self.assertLess(float(perceptual[0,0,0,0]),0); self.assertGreater(float(perceptual[0,1,0,0]),0)

    def test_calibration_falls_back_when_no_policy_improves(self):
        record={"gain":torch.ones(1,1,1),"harm":torch.zeros(1,1,1),"delta":-torch.ones(1,1,1),"local_mse":torch.tensor([[[.01]],[[.02]]]),"local_perceptual":torch.tensor([[[.01]],[[.01]]])}
        config=json.loads((ROOT/"configs/foundir_pairwise_safe_candidate_gate_noise_v1.json").read_text())
        policy,baseline=select_calibration_policy([record],config)
        self.assertIsNone(policy); self.assertGreater(baseline["psnr"],0)


if __name__ == "__main__":
    unittest.main()
