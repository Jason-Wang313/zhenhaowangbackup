from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lowlight_restoration.rain_full_data import RainSourceSchedule, RainFullDataset, crop_pair
from smoke import train_foundir_rain_flux2_precision_full_v1_ddp as full


def rows_for_schedule():
    return [{'role': 'direct_endpoint_train', 'split': 'train',
             'source_group': f'g{group}', 'record_id': f'r{group}_{i}'}
            for group in range(7) for i in range(group + 1)]


class FullRainTests(unittest.TestCase):
    def test_source_balance_record_cycle_and_rank_partition(self):
        rows = rows_for_schedule()
        schedule = RainSourceSchedule(rows, 123)
        reverse = list(reversed(rows))
        other = RainSourceSchedule(reverse, 123)
        for epoch in range(7):
            chosen = [rows[schedule.index(epoch * 7 + i)] for i in range(7)]
            self.assertEqual(len({r['source_group'] for r in chosen}), 7)
        chosen = [rows[schedule.index(i)] for i in range(49)]
        for group in range(7):
            records = [r['record_id'] for r in chosen if r['source_group'] == f'g{group}']
            self.assertEqual(len(set(records[:group + 1])), group + 1)
        self.assertEqual([r['record_id'] for r in chosen],
                         [reverse[other.index(i)]['record_id'] for i in range(49)])
        positions = [micro * 8 + rank for rank in range(8) for micro in range(4)]
        self.assertEqual(sorted(positions), list(range(32)))
        rows[0]['split'] = 'test'
        with self.assertRaises(ValueError):
            RainSourceSchedule(rows, 123)

    def make_row(self, directory, width=520, height=400):
        array = np.random.default_rng(11).integers(0, 256, (height, width, 3), dtype=np.uint8)
        path = Path(directory) / 'pair.png'
        Image.fromarray(array).save(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return {'role': 'direct_endpoint_train', 'split': 'train', 'record_id': 'r0',
                'source_group': 'g0', 'width': str(width), 'height': str(height),
                'lq_path': path.name, 'gt_path': path.name, 'lq_sha256': sha, 'gt_sha256': sha}

    def test_aligned_random_crop_reflect_padding_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.make_row(directory)
            verified = {}
            pair = crop_pair(directory, row, crop_size=512, position=0, seed=17, verified=verified)
            repeat = crop_pair(directory, row, crop_size=512, position=0, seed=17, verified=verified)
            different = crop_pair(directory, row, crop_size=512, position=1, seed=17, verified=verified)
            self.assertEqual(tuple(pair[0].shape), (3, 512, 512))
            self.assertTrue(torch.equal(*pair))
            self.assertTrue(torch.equal(pair[0], repeat[0]))
            self.assertFalse(torch.equal(pair[0], different[0]))
            self.assertLessEqual(float(pair[0].max()), 1)
            self.assertGreaterEqual(float(pair[0].min()), -1)
            with (Path(directory) / row['lq_path']).open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                crop_pair(directory, row, crop_size=512, position=0, seed=17, verified=verified)

    def test_spawn_loader_and_no_decoded_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.make_row(directory)
            data = RainFullDataset(directory, [row], steps=1, rank=3, world=8, seed=17)
            loader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=2,
                prefetch_factor=2, multiprocessing_context='spawn')
            batches = list(loader)
            self.assertEqual([int(b['position'][0]) for b in batches], [3, 11, 19, 27])
            self.assertTrue(all(torch.equal(b['lq'], b['gt']) for b in batches))
            self.assertNotIn('lq', row)
            self.assertNotIn('gt', row)

    def test_full_contract_and_frozen_development(self):
        config, parent = full.read_contract(full.ROOT / 'configs/foundir_rain_flux2_precision_full_v1.json')
        training, probe, development = full.load_protocol(config, parent)
        self.assertEqual((len(training), len(probe), len(development)), (38334, 32, 100))
        self.assertEqual(len({r['source_group'] for r in training}), 5780)
        self.assertFalse({r['source_group'] for r in training} & {r['source_group'] for r in development})
        self.assertEqual(full.digest(development), '1652d4e46634c1ee248e1984ea2aeaec271d8491a67dcc3acf12838cb181d783')
        self.assertEqual(full.digest(probe), 'cfd29a617d50b25c87ba8d3e246566cd789fae412b057a677d2a6c43afdd969a')
        schedule = RainSourceSchedule(training, config['training']['seed'])
        self.assertEqual(len({training[schedule.index(i)]['source_group'] for i in range(5780)}), 5780)


if __name__ == '__main__':
    unittest.main()
