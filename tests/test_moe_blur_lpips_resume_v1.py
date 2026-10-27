import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from lowlight_restoration.rain_blur_moe_data import task_position
from smoke import moe_blur_lpips_resume_v1 as migration
from tests.test_moe_blur_lpips_continuation_v1 import ContinuationTests as Fixtures


class MigrationTests(unittest.TestCase):
    def test_config_fixed_offset_and_hardware_support(self):
        cfg = migration.read_migration()
        self.assertEqual(cfg['start_step'], 10500)
        self.assertEqual(cfg['next_sample_position'], 336000)
        self.assertEqual(cfg['end_step']-cfg['start_step'], 500)
        self.assertEqual(cfg['homogeneous_gpu_families'], ['H100', 'H200'])
        self.assertEqual(cfg['save_steps'], [10600,10700,10800,10900,11000])

    def test_prefix_hash_matches_uninterrupted_sequence(self):
        class Schedule:
            def index(self, position): return position % 3
        training = {task: [{'record_id': task+str(i)} for i in range(3)] for task in ('rain','blur')}
        for rank in range(8):
            dataset = SimpleNamespace(rank=rank, world=8, tasks=['rain','blur'], training=training,
                                      schedules={t: Schedule() for t in training})
            full, resumed = hashlib.sha256(), hashlib.sha256()
            migration.reconstruct_trace_prefix(dataset, resumed, 10500)
            for index in range(10000*4, 11000*4):
                task, position = task_position(index, rank, 8)
                row = training[dataset.tasks[task]][position%3]
                migration.trace_event(full, index*8+rank, task, row['record_id'])
                if index >= 10500*4:
                    migration.trace_event(resumed, index*8+rank, task, row['record_id'])
            self.assertEqual(full.hexdigest(), resumed.hexdigest())

    def fixture(self, gpu):
        control, resumed = Fixtures().fixture_results()
        resumed.update(start_step=10500, optimizer_steps_this_run=500, this_launch_presentations=16000,
                       original_pair_source_sha256=control['source_sha256'], source_sha256={'new': 'sources'},
                       gpu_names=[gpu]*8, hardware={'hostname': 'new', 'gpu_uuids': ['new']},
                       hardware_history=[{'old': 'H100'}, {'new': gpu}])
        return control, resumed

    def test_cross_hardware_comparison_is_explicit_not_strict_same_node(self):
        for gpu in ('H100', 'H200'):
            a,b = self.fixture(gpu)
            result = migration.compare_migrated(a,b)
            self.assertFalse(result['same_physical_node_for_full_pair'])
            self.assertEqual(result['same_GPU_model_after_migration'], gpu=='H100')
            self.assertTrue(result['predeclared_repair_signal'])
            self.assertFalse(result['absolute_joint_gate_pass']['blur_lpips4x'])
            self.assertFalse(result['VLM_advantage_or_novelty_established'])

    def test_migration_still_rejects_changed_scientific_protocol(self):
        a,b = self.fixture('H200')
        for key,bad in [('config_sha256','wrong'), ('original_pair_source_sha256',{}),
                        ('sample_trace_sha256_by_rank',['wrong']), ('steps',10750),
                        ('start_step',10000)]:
            wrong=copy.deepcopy(b); wrong[key]=bad
            with self.assertRaises(ValueError): migration.compare_migrated(a,wrong)

    def test_unchanged_old_sources_and_bound_resume_metadata(self):
        from smoke.train_foundir_moe_blur_lpips_continuation_v1_ddp import source_hashes
        cfg,receipt,old=migration.verify_migration(source_hashes(), artifacts=False)
        self.assertEqual(receipt['arm'],'blur_lpips4x')
        self.assertEqual(receipt['next_sample_position'],336000)

    def test_launcher_requires_stop_ack_and_eight_gpus(self):
        text=(migration.ROOT/'smoke/run_foundir_moe_blur_lpips_resume_v1_8gpu.sh').read_text()
        self.assertIn('OLD_MOE_STOPPED',text)
        self.assertIn('--nproc_per_node=8',text)
        self.assertNotIn('--nnodes',text)
        self.assertNotIn('start_foundir_novel_ppt',text)


if __name__ == '__main__': unittest.main()
