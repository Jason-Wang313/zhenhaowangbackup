import copy
import csv
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from smoke import existing_flux_common_protocol_contract_v1 as p
from smoke import existing_flux_replay_contract_v1 as c


class ProtocolContractTests(unittest.TestCase):
    def test_parent_artifact_hash_and_size_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.json"
            artifact.write_bytes(b"parent evidence\n")
            expected = {"sha256": c.sha(artifact), "bytes": artifact.stat().st_size}
            report = {"run_id": "parent", "artifact_hashes": {artifact.name: expected}}
            with patch.object(c, "load_receipt", return_value={"verified": True}) as loader:
                self.assertEqual(p.check_parent(root, report), {"verified": True})
                loader.assert_called_once()
                loader.reset_mock()
                for bad in ({**expected, "sha256": "bad"},
                            {**expected, "bytes": expected["bytes"] + 1},
                            {**expected, "bytes": str(expected["bytes"])},
                            {"bytes": expected["bytes"]}):
                    with self.subTest(receipt=bad), self.assertRaises(ValueError):
                        p.check_parent(root, {**report, "artifact_hashes": {artifact.name: bad}})
                loader.assert_not_called()
                self.assertEqual(p.check_parent(root, {**report, "artifact_hashes": {
                    artifact.name: expected["sha256"]}}), {"verified": True})

    def test_config_keeps_scope_and_topology(self):
        cfg=c.read(c.ROOT/p.CONFIG)
        self.assertEqual(cfg["topology"],c.TOPOLOGY)
        self.assertEqual(cfg["gpu_family"],"H200")
        self.assertEqual(cfg["steps"],50)
        self.assertEqual(tuple(cfg["arms"]),p.ARMS)
        for key in ("training","router","automatic_follow_on","per_image_gt_selection"):
            self.assertFalse(cfg[key])
        self.assertEqual(cfg["records_per_task"]*len(cfg["tasks"]),128)

    def test_output_root_is_separate(self):
        self.assertNotEqual(p.run_path(Path("/tmp"),"same"),c.run_path(Path("/tmp"),"same"))
        with self.assertRaises(ValueError):
            p.run_path(Path("/tmp"),"../bad")

    def test_gt_mapping_train_only_identical_to_parent(self):
        row={"record_id":"r1","split":"train","category":"01Blur","source_group":"source",
             "gt_sha256":"gt","lq_sha256":"lq","lq_path":"lq","width":"700","height":"800",
             "gt_path":"data/foundir/MillionIRData/Train/01Blur/GT/1.png"}
        parent={k:v for k,v in row.items() if k!="gt_path"}
        parent.update(crop_xy=[1,2],inference_seed=3)
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/"data.csv"
            def write(value):
                with file.open("w",newline="") as f:
                    writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(value)
            write([row])
            result=p.select_gt(file,{"blur":[parent]})
            self.assertEqual(result,{"r1":{"path":row["gt_path"],"sha256":"gt"}})
            for bad in ({**row,"split":"validation"},{**row,"gt_path":"data/test/GT/x.png"},
                        {**row,"gt_sha256":"changed"},{**row,"gt_path":row["gt_path"]+"/../bad"}):
                write([bad])
                with self.assertRaises(ValueError):
                    p.select_gt(file,{"blur":[parent]})
            write([row,row])
            with self.assertRaises(ValueError):
                p.select_gt(file,{"blur":[parent]})
            write([])
            with self.assertRaises(ValueError):
                p.select_gt(file,{"blur":[parent]})

    def test_aggregate_preserves_each_arm_and_reports_harm(self):
        rows=[]
        for i in range(2):
            row={"identity":{"key":str(i),"task":"noise"},
                 "metrics":{a:{"psnr":30.,"lpips":.1} for a in p.ARMS}}
            row["metrics"][p.ARMS[1]]["psnr"]-=1
            row["metrics"][p.ARMS[2]]["lpips"]+=.01
            row["metrics"][p.ARMS[3]]={"psnr":31.,"lpips":.09}
            rows.append(row)
        tolerance={"psnr_db":1e-6,"lpips":1e-8}
        result=p.aggregate(rows,["0","1"],tolerance)["noise"]
        self.assertEqual(result["deltas_vs_original"][p.ARMS[1]]["psnr"],-1)
        self.assertFalse(result["descriptive_mean_nonregression"][p.ARMS[1]])
        self.assertFalse(result["descriptive_mean_nonregression"][p.ARMS[2]])
        self.assertTrue(result["descriptive_mean_nonregression"][p.ARMS[3]])
        self.assertEqual(result["per_arm_harmed_images"][p.ARMS[1]],2)
        for invalid in (rows[:1],[rows[0],rows[0]]):
            with self.assertRaises(ValueError):
                p.aggregate(invalid,["0","1"],tolerance)
        bad=copy.deepcopy(rows);bad[0]["metrics"][p.ARMS[1]]["psnr"]=float("nan")
        with self.assertRaises(ValueError):
            p.aggregate(bad,["0","1"],tolerance)
        bad=copy.deepcopy(rows);bad[0]["metrics"].pop(p.ARMS[1])
        with self.assertRaises(ValueError):
            p.aggregate(bad,["0","1"],tolerance)

    def test_common_prompt_has_no_sample_label(self):
        from smoke.diagnose_foundir_existing_flux_common_protocol_v1 import prompt_view
        cfg=c.read(c.ROOT/p.CONFIG)
        view=prompt_view({"config":cfg})
        self.assertEqual(view["config"]["tasks"],["common"])
        self.assertEqual(view["experts"]["common"]["prompt"],cfg["common_prompt"])

    def test_parent_terminal_is_fixed(self):
        cfg=c.read(c.ROOT/p.CONFIG)
        report=p.terminal(c.ROOT,cfg)
        self.assertEqual(report["full_result"]["records"],128)
        with self.assertRaises(ValueError):
            p.terminal(c.ROOT,{**cfg,"parent_report_sha256":"bad"})

    def test_launcher_is_four_nodes_and_separate_from_parent(self):
        text=(c.ROOT/"smoke/run_foundir_existing_flux_common_protocol_v1_gpu.sh").read_text()
        self.assertIn("--nnodes=4 --nproc_per_node=2",text)
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1",text)
        self.assertIn("WALL_CAP=3600",text)
        self.assertIn("diagnose_foundir_existing_flux_common_protocol_v1.py",text)
        self.assertIn("existing_flux_common_protocol_cluster_v1",text)
        self.assertNotIn("--standalone",text)
        self.assertNotIn("replay_foundir_existing_flux_experts_v1.py",text)

    def test_metric_implementation_must_match_across_nodes(self):
        from smoke.diagnose_foundir_existing_flux_common_protocol_v1 import check_metric_runtimes
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for node in range(4):
                c.write_new(root/f"runtime_node{node}.json",{"metrics":{"version":"same"}})
            self.assertEqual(check_metric_runtimes(root),{"version":"same"})
            (root/"runtime_node3.json").write_text(json.dumps({"metrics":{"version":"other"}}))
            with self.assertRaises(ValueError):
                check_metric_runtimes(root)


class ProtocolTensorTests(unittest.TestCase):
    def test_four_output_arms_change_only_the_named_factor(self):
        import torch
        from smoke import diagnose_foundir_existing_flux_common_protocol_v1 as runner
        torch.manual_seed(7)
        lq,a,b=[torch.rand(1,3,8,8)*2-1 for _ in range(3)]
        for task,alpha in (("noise",.3),("blur",.75),("lowlight",.925),("haze",1.)):
            original={"decoded":a,"uint8":runner.replay.quantize(task,lq,a,alpha)}
            common={"decoded":b,"uint8":runner.replay.quantize(task,lq,b,alpha)}
            arms=runner.output_arms(task,lq,original,common,alpha)
            self.assertTrue(torch.equal(arms[p.ARMS[0]],original["uint8"]))
            self.assertTrue(torch.equal(arms[p.ARMS[1]],common["uint8"]))
            self.assertTrue(torch.equal(arms[p.ARMS[2]],runner.replay.quantize("haze",lq,a,1.)))
            self.assertTrue(torch.equal(arms[p.ARMS[3]],runner.replay.quantize("haze",lq,b,1.)))
            damaged={**original,"uint8":torch.zeros_like(original["uint8"])}
            with self.assertRaises(ValueError):
                runner.output_arms(task,lq,damaged,common,alpha)

    def test_h200_only_scope(self):
        import torch
        from smoke import diagnose_foundir_existing_flux_common_protocol_v1 as runner
        with patch.object(runner.replay,"hardware",return_value={"gpu_name":"NVIDIA H200"}), patch.object(
                torch.cuda,"get_device_properties",return_value=types.SimpleNamespace(total_memory=140*1024**3)):
            self.assertEqual(runner.hardware(False)["gpu_name"],"NVIDIA H200")
        with patch.object(runner.replay,"hardware",return_value={"gpu_name":"NVIDIA H100"}):
            with self.assertRaises(ValueError):
                runner.hardware(False)

    def test_training_gt_crop_and_hash_guard(self):
        import numpy as np
        from PIL import Image
        from smoke import diagnose_foundir_existing_flux_common_protocol_v1 as runner
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            rel="data/foundir/MillionIRData/Train/01Blur/GT/x.png"
            file=root/rel;file.parent.mkdir(parents=True)
            data=np.zeros((514,515,3),dtype=np.uint8);data[1,2]=255
            Image.fromarray(data).save(file)
            digest=c.sha(file)
            receipt={"gt":{"r":{"path":rel,"sha256":digest}}}
            row={"record_id":"r","split":"train","category":"01Blur","gt_sha256":digest,
                 "width":"515","height":"514","crop_xy":[2,1]}
            with patch.object(runner,"ROOT",root):
                image=runner.load_gt(receipt,row,"cpu")
                self.assertEqual(tuple(image.shape),(1,3,512,512))
                self.assertEqual(float(image[0,:,0,0].mean()),1.)
                with self.assertRaises(ValueError):
                    runner.load_gt(receipt,{**row,"split":"validation"},"cpu")
                file.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    runner.load_gt(receipt,row,"cpu")


if __name__=="__main__":
    unittest.main()
