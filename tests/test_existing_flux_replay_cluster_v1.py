import copy
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from smoke import existing_flux_replay_contract_v1 as c
from smoke import existing_flux_replay_cluster_v1 as cluster


class ClusterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        c.write_new(self.root / "receipt.json", {"world":8, "topology":c.TOPOLOGY})
        for node in range(4):
            c.write_new(self.root / f"runtime_node{node}.json", {"gpu":"H100" if node % 2 else "H200"})

    def boot(self, launch, node, master="node0", host=None):
        with patch.dict(os.environ, REPLAY_MASTER_ADDR=master, REPLAY_MASTER_PORT="29681"), patch(
                "socket.gethostname", return_value=host or f"node{node}"):
            cluster.publish(self.root, launch, node, "boot")

    def test_rank_mapping_and_complete_partition(self):
        seen = []
        tasks = list(c.CATEGORIES)
        rows = {t:list(range(32)) for t in tasks}
        for node in range(4):
            for local in range(2):
                rank, device = c.worker_rank(node, local, 8, 2, node*2+local)
                self.assertEqual(device, local)
                task, values = c.assigned(tasks, rows, rank, 8)
                seen.extend((task, i) for i in values)
        self.assertEqual(len(seen), 128)
        self.assertEqual(len(set(seen)), 128)
        for args in ((0,0,8,8,0), (1,0,8,2,0), (4,0,8,2,8), (0,2,8,2,2)):
            with self.assertRaises(ValueError):
                c.worker_rank(*args)

    def test_old_single_node_receipt_rejected(self):
        with self.assertRaisesRegex(ValueError, "topology"):
            c.validate_topology({"world":8})
        c.validate_topology(c.read(self.root / "receipt.json"))

    def test_record_binds_its_node_runtime(self):
        from smoke.replay_foundir_existing_flux_experts_v1 import identity_for
        a = identity_for("blur", {"record_id":"x"}, "r", "p", 0, "h100")
        b = identity_for("blur", {"record_id":"x"}, "r", "p", 0, "h200")
        with self.assertRaisesRegex(ValueError, "identity"):
            c.validate_record({"identity": a, "files": {}}, b, self.root)

    def test_four_nodes_complete_and_resume_new_attempt(self):
        for launch in ("attempt1", "resume2"):
            for node in range(4):
                self.boot(launch, node)
            self.assertTrue(cluster.check(self.root, launch, "boot"))
            cluster.publish(self.root, launch, 0, "prepared")
            self.assertTrue(cluster.check(self.root, launch, "prepared"))
            for event in ("ready", "done"):
                for node in range(3):
                    cluster.publish(self.root, launch, node, event)
                self.assertFalse(cluster.check(self.root, launch, event))
                cluster.publish(self.root, launch, 3, event)
                self.assertTrue(cluster.check(self.root, launch, event))
            c.write_new(self.root / "result.json", {"pass":True})
            cluster.publish(self.root, launch, 0, "verified")
            self.assertTrue(cluster.check(self.root, launch, "verified"))

    def test_duplicate_attempt_rejected(self):
        self.boot("a", 0)
        with self.assertRaises(FileExistsError):
            self.boot("a", 0)

    def test_stale_markers_and_failure_not_success(self):
        for node in range(4):
            cluster.publish(self.root, "old", node, "done")
        self.assertFalse(cluster.check(self.root, "new", "done"))
        folder = cluster.directory(self.root, "new")
        c.write_new(folder / "done_node0.json", c.read(cluster.directory(self.root,"old") / "done_node0.json"))
        with self.assertRaisesRegex(ValueError, "stale"):
            cluster.check(self.root, "new", "done")
        cluster.publish(self.root, "broken", 2, "failed")
        with self.assertRaises(RuntimeError):
            cluster.check(self.root, "broken", "done")

    def test_wrong_master_or_duplicate_node_rejected(self):
        for node in range(4):
            self.boot("address", node, master="other" if node == 3 else "node0")
            self.boot("host", node, host="same")
        for launch in ("address", "host"):
            with self.assertRaises(ValueError):
                cluster.check(self.root, launch, "boot")

    def test_only_node0_publishes_common_markers(self):
        for event in ("prepared", "verified"):
            with self.assertRaises(ValueError):
                cluster.publish(self.root, "a", 1, event)

    def test_shell_step_preserves_status_and_stops_on_peer_failure(self):
        script = (c.ROOT / "smoke/run_foundir_existing_flux_replay_v1_gpu.sh").read_text()
        function = "step() {" + script.split("step() {", 1)[1].split("\n}", 1)[0] + "\n}"
        setup = 'set -euo pipefail; START=$SECONDS; WALL_CAP=20; OUTPUT="$1"; LAUNCH_ID=a;\n'
        for command, expected in (("true",0), ("false",1)):
            result = subprocess.run(["bash","-c",setup+function+f"\nstep {command}","test",str(self.root)],
                                    capture_output=True,timeout=10)
            self.assertEqual(result.returncode,expected)
        cluster.publish(self.root,"a",2,"failed")
        result = subprocess.run(["bash","-c",setup+function+"\nstep sleep 60","test",str(self.root)],
                                capture_output=True,timeout=10)
        self.assertEqual(result.returncode,1)
        self.assertIn(b"another node failed",result.stdout)

    def test_runtime_or_receipt_changes_rejected(self):
        for node in range(4):
            cluster.publish(self.root, "a", node, "done")
        (self.root / "runtime_node0.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "runtime"):
            cluster.check(self.root, "a", "done")
        (self.root / "receipt.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "receipt"):
            cluster.check(self.root, "a", "done")

    def test_two_local_gpus_and_per_node_family(self):
        from smoke.replay_foundir_existing_flux_experts_v1 import hardware
        import torch
        def gpu(name):
            return types.SimpleNamespace(name=name, total_memory=80*1024**3)
        for names in (("NVIDIA H100",)*2, ("NVIDIA H200",)*2):
            with patch.object(torch.cuda,"device_count",return_value=2), patch.object(
                    torch.cuda,"get_device_properties",side_effect=[gpu(n) for n in names]), patch.object(
                    Path,"read_text",return_value="driver"), patch.dict(
                    sys.modules, {"transformers":types.SimpleNamespace(__version__="mock-policy-test")}):
                self.assertEqual(hardware(8,require_idle=False)["local_gpus"],2)
        with patch.object(torch.cuda,"device_count",return_value=8):
            with self.assertRaises(ValueError):
                hardware(8,require_idle=False)
        with patch.object(torch.cuda,"device_count",return_value=2), patch.object(
                torch.cuda,"get_device_properties",side_effect=[gpu("NVIDIA H100"),gpu("NVIDIA H200")]):
            with self.assertRaises(ValueError):
                hardware(8,require_idle=False)


if __name__ == "__main__":
    unittest.main()
