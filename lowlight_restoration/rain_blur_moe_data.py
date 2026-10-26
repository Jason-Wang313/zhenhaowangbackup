"""Two-task source-isolated data, preserving original development crops."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from torch.utils.data import Dataset

from lowlight_restoration.rain_full_data import RainSourceSchedule, crop_pair


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_protocol(root, rain_training, rain_development, manifest):
    with (Path(root) / manifest).open(newline='') as handle:
        blur = [dict(r) for r in csv.DictReader(handle)
                if r['category'] == '01Blur' and r['composition_size'] == '1']
    original_blur_train = [r for r in blur if r['split'] == 'train']
    blur_dev = sorted([r for r in blur if r['split'] == 'validation'], key=lambda r: r['record_id'])
    if (len(original_blur_train), len(blur_dev)) != (105057, 2595):
        raise RuntimeError('Blur pool identity changed')
    blocked_groups = {r['source_group'] for r in rain_development + blur_dev}
    blocked_gt = {r['gt_sha256'] for r in rain_development + blur_dev}
    training = {}
    excluded = {}
    for task, rows in [('rain', rain_training), ('blur', original_blur_train)]:
        allowed = []
        for source in rows:
            if source['split'] != 'train':
                raise RuntimeError('non-training row')
            if source['source_group'] in blocked_groups or source['gt_sha256'] in blocked_gt:
                continue
            row = dict(source)
            row['role'] = 'direct_endpoint_train'
            if min(int(row['width']), int(row['height'])) < 256:
                raise RuntimeError('unsupported training geometry')
            allowed.append(row)
        training[task] = sorted(allowed, key=lambda r: r['record_id'])
        excluded[task] = len(rows) - len(allowed)
    for row in blur_dev:
        width, height = int(row['width']), int(row['height'])
        if min(width, height) < 512:
            raise RuntimeError('Blur development crop would require resizing')
        raw = hashlib.sha256(f"2026091111:full_development_crop:{row['record_id']}".encode()).digest()
        row['crop_y'] = str(int.from_bytes(raw[:8], 'big') % (height - 512 + 1))
        row['crop_x'] = str(int.from_bytes(raw[8:16], 'big') % (width - 512 + 1))
        row['role'] = 'blur_development'
    # A source-balanced preview, selected WITHOUT pixels or quality scores.
    representatives = {}
    for row in sorted(blur_dev, key=lambda r: hashlib.sha256(r['record_id'].encode()).hexdigest()):
        representatives.setdefault(row['source_group'], row)
    preview = list(representatives.values())[:100]
    development = {'rain': rain_development, 'blur_preview': preview, 'blur_full': blur_dev}
    metadata = {'train_records': {k: len(v) for k, v in training.items()},
                'train_sources': {k: len({r['source_group'] for r in v}) for k, v in training.items()},
                'excluded_cross_task_development_records': excluded,
                'training_digest': {k: digest(v) for k, v in training.items()},
                'development_digest': {k: digest(v) for k, v in development.items()},
                'development_records': {k: len(v) for k, v in development.items()},
                'protected_pixels_accessed': False}
    return training, development, metadata


def task_position(index, rank, world):
    # Each global microbatch has one task; each optimizer update has 16/16.
    task = index % 2
    per_task_position = (index // 2) * world + rank
    return task, per_task_position


class RainBlurDataset(Dataset):
    def __init__(self, root, training, *, steps, rank, world=8, seed=2026102601):
        if world != 8 or not 0 <= rank < world:
            raise ValueError('requires eight ranks')
        self.root, self.training, self.steps = root, training, steps
        self.rank, self.world, self.seed = rank, world, seed
        self.tasks = ['rain', 'blur']
        self.schedules = {task: RainSourceSchedule(training[task], seed + i)
                          for i, task in enumerate(self.tasks)}
        self.verified = {}

    def __len__(self):
        return self.steps * 4

    def __getitem__(self, index):
        task_id, position = task_position(index, self.rank, self.world)
        task = self.tasks[task_id]
        row = self.training[task][self.schedules[task].index(position)]
        lq, gt = crop_pair(self.root, row, crop_size=512, position=position,
                          seed=self.seed + task_id, verified=self.verified)
        return {'lq': lq, 'gt': gt, 'task_id': task_id, 'record_id': row['record_id'],
                'source_group': row['source_group'], 'position': index * self.world + self.rank}
