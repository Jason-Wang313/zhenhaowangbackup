from __future__ import annotations

import unittest

from smoke.qwen_multidegradation_lora_preflight import ROOT, load_config
from smoke.select_qwen_multidegradation_lora import score, validation_candidates


def candidate_result(*, triple: float, combined: float) -> dict[str, object]:
    return {
        "variant": "merger_trainable",
        "validation_checkpoint_stage": "candidate",
        "metrics": {
            "combined_label_exact_match_rate": combined,
            "severity_typed_accuracy": 0.8,
            "degradation_set_exact_match_rate": 0.85,
            "type_macro_f1": 0.94,
            "severity_typed_mae_with_missing_penalty": 0.4,
            "by_composition_size": {
                "3": {"degradation_set_exact_match_rate": triple}
            },
        },
    }


class QwenMultiDegradationSelectionTests(unittest.TestCase):
    def test_continuation_prioritizes_triple_exact_before_other_metrics(self):
        stronger_triples = candidate_result(triple=0.81, combined=0.66)
        stronger_combined = candidate_result(triple=0.80, combined=0.90)
        self.assertGreater(score(stronger_triples), score(stronger_combined))

    def test_continuation_has_exact_epoch_five_six_eight_result_paths(self):
        config, _ = load_config(
            ROOT / "configs/qwen_multidegradation_per_type_rank32_continue_v2.json"
        )
        root = ROOT / config["runtime"]["output_root"]
        candidates = validation_candidates(config, root)
        self.assertEqual([name for name, _, _ in candidates], config["data"]["validation_candidate_stages"])
        self.assertTrue(str(candidates[0][2]).endswith("validation_result_H5_per_type_dialogue_epoch5.json"))
        self.assertTrue(str(candidates[1][2]).endswith("validation_result_H6_per_type_dialogue_epoch6.json"))
        self.assertTrue(str(candidates[2][2]).endswith("validation_result.json"))


if __name__ == "__main__":
    unittest.main()
