"""GT-free constant-offset synchronization of existing FLUX tiles.

Solve overlap mean differences once, with a gauge that preserves the unrounded
stitched image's RGB mean. No image/GT fitting, sweep, or output selection.
"""
import numpy as np


def synchronize(row, tiles, images):
    h, w = int(row['height']), int(row['width'])
    if not tiles or len({t['key'] for t in tiles}) != len(tiles):
        raise ValueError('unique nonempty tile list required')
    if set(images) != {t['key'] for t in tiles}:
        raise ValueError('tile population mismatch')
    total = np.zeros((h, w, 3), np.float32)
    norm = np.zeros((h, w), np.float32)
    windows = []
    for t in tiles:
        a = images[t['key']]
        if a.dtype != np.uint8 or a.shape != (512, 512, 3):
            raise ValueError('native uint8 512x512 RGB tiles required')
        x, y = t['x'], t['y']; th, tw = min(512, h-y), min(512, w-x)
        if min(x, y) < 0 or min(th, tw) <= 0:
            raise ValueError('invalid tile coordinates')
        vy = np.hanning(th).astype(np.float32) if th > 2 else np.ones(th, np.float32)
        vx = np.hanning(tw).astype(np.float32) if tw > 2 else np.ones(tw, np.float32)
        weight = np.maximum(np.sqrt(np.outer(vy, vx)), np.float32(1e-3))
        total[y:y+th, x:x+tw] += a[:th, :tw].astype(np.float32)*weight[..., None]
        norm[y:y+th, x:x+tw] += weight
        windows.append((x, y, th, tw, weight))
    if not (norm > 0).all():
        raise ValueError('uncovered pixels')
    base_float = total/norm[..., None]
    base = np.round(base_float).clip(0, 255).astype(np.uint8)
    by_xy = {(t['x'], t['y']): i for i, t in enumerate(tiles)}
    xs, ys = sorted({t['x'] for t in tiles}), sorted({t['y'] for t in tiles})
    if len(by_xy) != len(xs)*len(ys):
        raise ValueError('complete Cartesian tile grid required')
    edges = [(by_xy[x1, y], by_xy[x2, y]) for y in ys for x1, x2 in zip(xs, xs[1:])]
    edges += [(by_xy[x, y1], by_xy[x, y2]) for x in xs for y1, y2 in zip(ys, ys[1:])]
    n = len(tiles); lap = np.zeros((n, n)); rhs = np.zeros((n, 3)); edge_data = []
    for i, j in edges:
        a, b = tiles[i], tiles[j]
        x0, y0 = max(a['x'], b['x']), max(a['y'], b['y'])
        x1, y1 = min(a['x']+512, b['x']+512, w), min(a['y']+512, b['y']+512, h)
        if x1 <= x0 or y1 <= y0:
            raise ValueError('adjacent tiles must overlap')
        aa = images[a['key']][y0-a['y']:y1-a['y'], x0-a['x']:x1-a['x']].astype(np.float64)
        bb = images[b['key']][y0-b['y']:y1-b['y'], x0-b['x']:x1-b['x']].astype(np.float64)
        d = (aa-bb).mean((0, 1)); area = (x1-x0)*(y1-y0)
        edge_data.append((i, j, area, d))
    area_total = sum(e[2] for e in edge_data) or 1
    for i, j, area, d in edge_data:
        weight = area/area_total
        lap[i, i] += weight; lap[j, j] += weight
        lap[i, j] -= weight; lap[j, i] -= weight
        rhs[i] += weight*d; rhs[j] -= weight*d
    q = np.array([np.sum(weight.astype(np.float64)/norm[y:y+th, x:x+tw])/(h*w)
                  for x, y, th, tw, weight in windows])
    system = np.zeros((n+1, n+1)); system[:n, :n] = lap
    system[n, :n] = q; system[:n, n] = q
    target = np.zeros((n+1, 3)); target[:n] = rhs
    offsets = np.linalg.solve(system, target)[:n]
    field = np.zeros((h, w, 3), np.float64)
    for offset, (x, y, th, tw, weight) in zip(offsets, windows):
        field[y:y+th, x:x+tw] += (weight.astype(np.float64)/norm[y:y+th, x:x+tw])[..., None]*offset
    mean_shift = field.mean((0, 1))
    if not np.isfinite(field).all() or np.max(np.abs(mean_shift)) > 1e-7:
        raise ValueError('nonfinite correction or failed RGB mean preservation')
    corrected_float = base_float.astype(np.float64)-field
    corrected = np.round(corrected_float).clip(0, 255).astype(np.uint8)
    before = sum(area*float(np.mean(d*d)) for i, j, area, d in edge_data)/area_total
    after = sum(area*float(np.mean((d-offsets[i]+offsets[j])**2)) for i, j, area, d in edge_data)/area_total
    if after > before + 1e-8:
        raise ValueError('least-squares disagreement increased')
    stats = {'tiles': n, 'edges': len(edges), 'offsets_RGB_255': offsets.tolist(),
             'mean_correction_RGB_255_before_clip': mean_shift.tolist(),
             'correction_RMS_255': float(np.sqrt(np.mean(field*field))),
             'overlap_constant_MSE_255_before': before, 'overlap_constant_MSE_255_after': after,
             'new_out_of_range_fraction': float(np.mean((corrected_float < 0)|(corrected_float > 255))),
             'mean_RGB_change_255_after_quantization': (corrected.astype(np.float64)-base).mean((0, 1)).tolist(),
             'GT_input': False, 'global_mean_constraint': 'unrounded_before_clipping_only'}
    return corrected, base, stats
