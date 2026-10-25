"""Bounded-memory, deterministic source-balanced Rain pair loading."""
from __future__ import annotations

import hashlib
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RainSourceSchedule:
    def __init__(self, rows, seed):
        groups = {}
        for index, row in enumerate(rows):
            if row['role'] != 'direct_endpoint_train' or row['split'] != 'train':
                raise ValueError('training schedule rejects non-training rows')
            groups.setdefault(row['source_group'], []).append(index)
        self.groups = {key: sorted(value, key=lambda i: rows[i]['record_id'])
                       for key, value in sorted(groups.items())}
        self.names = list(self.groups)
        self.seed = seed
        if not self.names:
            raise ValueError('empty training data')

    @lru_cache(maxsize=4)
    def group_order(self, epoch):
        order = list(range(len(self.names)))
        random.Random(self.seed + epoch).shuffle(order)
        return order

    def index(self, position):
        epoch, offset = divmod(position, len(self.names))
        group = self.names[self.group_order(epoch)[offset]]
        candidates = self.groups[group]
        cycle, occurrence = divmod(epoch, len(candidates))
        order = list(range(len(candidates)))
        key = int(hashlib.sha256(f'{self.seed}:{group}:{cycle}'.encode()).hexdigest()[:16], 16)
        random.Random(key).shuffle(order)
        return candidates[order[occurrence]]


def crop_pair(root, row, *, crop_size, position, seed, verified):
    """Verify bytes before first decode; no persistent decoded-image cache."""
    rng = random.Random(seed + 1000003 * (position + 1))
    width, height = int(row['width']), int(row['height'])
    x = rng.randrange(max(width - crop_size, 0) + 1)
    y = rng.randrange(max(height - crop_size, 0) + 1)
    code = rng.randrange(8)
    tensors = []
    for side in ('lq', 'gt'):
        path = Path(root) / row[side + '_path']
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        key = str(path)
        if verified.get(key) != signature:
            hasher = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    hasher.update(block)
            if hasher.hexdigest() != row[side + '_sha256']:
                raise RuntimeError(f'image identity changed: {row["record_id"]} {side}')
            verified[key] = signature
        with Image.open(path) as image:
            if image.size != (width, height):
                raise RuntimeError(f'image geometry changed: {row["record_id"]}')
            array = np.array(image.convert('RGB').crop(
                (x, y, min(x + crop_size, width), min(y + crop_size, height))), copy=True)
        pad_y, pad_x = crop_size - array.shape[0], crop_size - array.shape[1]
        if pad_y or pad_x:
            array = np.pad(array, ((0, pad_y), (0, pad_x), (0, 0)), mode='reflect')
        array = np.rot90(array, code % 4)
        if code >= 4:
            array = np.flip(array, axis=1)
        tensors.append(torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1))
    return tensors[0], tensors[1]


class RainFullDataset(Dataset):
    def __init__(self, root, rows, *, steps, rank, world, seed, crop_size=512):
        self.root, self.rows = root, rows
        self.steps, self.rank, self.world = steps, rank, world
        self.seed, self.crop_size = seed, crop_size
        self.schedule = RainSourceSchedule(rows, seed)
        self.verified = {}

    def __len__(self):
        return self.steps * 4

    def __getitem__(self, index):
        position = index * self.world + self.rank
        row = self.rows[self.schedule.index(position)]
        lq, gt = crop_pair(self.root, row, crop_size=self.crop_size,
                          position=position, seed=self.seed, verified=self.verified)
        return {'lq': lq, 'gt': gt, 'record_id': row['record_id'],
                'source_group': row['source_group'], 'position': position}
