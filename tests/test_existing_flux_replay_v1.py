import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from smoke import existing_flux_replay_contract_v1 as c


class ContractTests(unittest.TestCase):
    def test_shards_cover_identical_128_records(self):
        tasks = list(c.CATEGORIES)
        rows = {t: list(range(32)) for t in tasks}
        for world in (4, 8):
            seen = []
            for rank in range(world):
                task, sub = c.assigned(tasks, rows, rank, world)
                self.assertEqual(len(sub), 128 // world)
                seen.extend((task, i) for i in sub)
            self.assertEqual(len(seen), len(set(seen)))
            self.assertEqual(set(seen), {(t,i) for t in tasks for i in range(32)})

    def test_invalid_world_and_rank(self):
        for world, rank in ((2, 0), (4, 4), (8, -1)):
            with self.assertRaises(ValueError):
                c.assigned(list(c.CATEGORIES), {}, rank, world)

    def test_immutable_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "record.json"
            c.write_new(p, {"pass": True})
            c.write_new(p, {"pass": True})
            with self.assertRaises(ValueError):
                c.write_new(p, {"pass": False})
            self.assertEqual(c.read(p), {"pass": True})

    def test_path_and_id_escape(self):
        with self.assertRaises(ValueError):
            c.path(Path("/tmp/replay_test"), "../escape")
        for run_id in ("../bad", "x/y", "", "x"*81, "a;touch bad"):
            with self.assertRaises(ValueError):
                c.run_path(Path("/tmp"), run_id)

    def test_corrupted_tensor_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = root / "tensor"
            p.write_bytes(b"a")
            info = {"identity": {"key":"x"}, "files":{"tensor":c.sha(p)}}
            c.validate_record(info, {"key":"x"}, root)
            p.write_bytes(b"b")
            with self.assertRaises(ValueError):
                c.validate_record(info, {"key":"x"}, root)

    def test_incomplete_failed_and_nonfinite_summary(self):
        record = {"identity":{"key":"a"}, "max_abs_rgb":0., "uint8_equal":True, "pass":True}
        self.assertTrue(c.validate_summary([record], ["a"], 1e-6))
        bad = {**record, "max_abs_rgb":0.01, "pass":False}
        self.assertFalse(c.validate_summary([bad], ["a"], 1e-6))
        for values, expected in (([], ["a"]), ([record, record],["a"]), ([record],["b"])):
            with self.assertRaises(ValueError):
                c.validate_summary(values, expected, 1e-6)
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                c.validate_summary([{**record,"max_abs_rgb":value}], ["a"], 1e-6)
        with self.assertRaises(ValueError):
            c.validate_summary([{**record,"uint8_equal":False}], ["a"], 1e-6)

    def test_selection_excludes_all_nontrain_sources_and_gt(self):
        config = {"tasks":["blur"], "records_per_task":2, "crop_size":512, "selection_seed":11}
        expert = {"blur":{"data_protocol":{"crop_seed":7}, "inference":{"seed":9, "seed_namespace":"original"}}}
        def row(i, split="train", source=None, gt=None):
            return {"record_id":f"r{i}","category":"01Blur","composition_size":"1","split":split,
                    "source_group":source or f"{i:064x}", "gt_sha256":gt or f"{i+100:064x}",
                    "lq_sha256":f"{i+200:064x}", "lq_path":f"data/foundir/MillionIRData/Train/01Blur/LQ/{i}.png",
                    "gt_path":"DO_NOT_OPEN", "width":"800", "height":"700"}
        rows = [row(1),row(2),row(3),row(4),row(5,"validation",source=f"{1:064x}"),
                row(6,"validation",gt=f"{102:064x}")]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.csv"
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            selected = c.select_rows(p, config, expert)["blur"]
            self.assertEqual({r["record_id"] for r in selected},{"r3","r4"})
            self.assertTrue(all("gt_path" not in r for r in selected))
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(reversed(rows))
            self.assertEqual(selected, c.select_rows(p, config, expert)["blur"])

    def test_resume_world_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c.write_new(root / "receipt.json", {"world":8})
            with self.assertRaisesRegex(ValueError, "world"):
                c.load_receipt(root, root, 4)

    def test_hardware_policy_and_no_training(self):
        config = c.read(c.ROOT / c.CONFIG)
        self.assertEqual(config["world_sizes"], [8])
        self.assertEqual(config["topology"], c.TOPOLOGY)
        self.assertFalse(config["training"])
        self.assertFalse(config["read_gt"])
        self.assertFalse(config["automatic_follow_on"])
        for world in (8,):
            self.assertEqual(config["wall_cap_seconds"][str(world)] * world, 57600)


class ImageMathTests(unittest.TestCase):
    def test_quantization_preserves_original_two_stage_order(self):
        import torch
        from smoke.replay_foundir_existing_flux_experts_v1 import quantize
        from lowlight_restoration.scalar_residual import quantized_scalar_residual
        torch.manual_seed(1)
        lq, raw = [torch.rand(1,3,16,16)*2-1 for _ in range(2)]
        first, first_q = quantized_scalar_residual(lq, raw, 1.)
        self.assertTrue(torch.equal(quantize("haze",lq,raw,1.), first))
        for task, alpha in (("blur",.75),("noise",.3),("lowlight",.925)):
            wanted = quantized_scalar_residual(lq, first_q, alpha)[0]
            self.assertTrue(torch.equal(quantize(task,lq,raw,alpha), wanted))

    def test_verifier_recomputes_saved_tensors(self):
        import torch
        from smoke.replay_foundir_existing_flux_experts_v1 import identity_for, save_tensors, verify
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            tasks = list(c.CATEGORIES)
            receipt = {"world":4, "config":{"tasks":tasks,"rgb_max_abs_tolerance":1e-6},
                       "rows":{t:[{"record_id":"train1"}] for t in tasks},
                       "experts":{t:{"prompt":t} for t in tasks}}
            c.write_new(output / "receipt.json", receipt)
            receipt_sha = c.sha(output / "receipt.json")
            prompt_tensor = output / "prompt.safetensors"
            prompt_tensor.write_bytes(b"mock-prompt-identity-only")
            c.write_new(output / "prompts.json", {"receipt_sha256":receipt_sha,
                "prompts":{t:t for t in tasks}, "tensors":{t:{"path":prompt_tensor.name,"sha256":c.sha(prompt_tensor)} for t in tasks}})
            prompts_sha = c.sha(output / "prompts.json")
            tensors = {"decoded":torch.zeros(1,3,512,512), "uint8":torch.full((1,3,512,512),128,dtype=torch.uint8)}
            for rank, task in enumerate(tasks):
                ident = identity_for(task, receipt["rows"][task][0], receipt_sha, prompts_sha, rank)
                name, digest = save_tensors(output, task, tensors)
                original = {"identity":ident,"files":{name:digest},"tensor":name}
                original_path = output / "original" / f"{ident['key']}.json"
                c.write_new(original_path, original)
                bank = {**original,"original_receipt_sha256":c.sha(original_path),
                        "max_abs_rgb":0.,"uint8_equal":True,"pass":True}
                c.write_new(output / "bank" / f"{ident['key']}.json", bank)
            self.assertEqual(verify(output, receipt), 0)
            self.assertTrue(c.read(output / "result.json")["pass"])
            target = output / "bank" / "blur_train1.json"
            broken = c.read(target); broken["max_abs_rgb"] = .01; broken["pass"] = False
            target.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "reported comparison"):
                verify(output, receipt)


if __name__ == "__main__":
    unittest.main()
