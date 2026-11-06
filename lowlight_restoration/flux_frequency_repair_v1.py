"""Six train-fitted DCT bands on a frozen FLUX residual, without GT at inference."""
import numpy as np
from scipy.fft import dctn, idctn

BANDS = ('dc', 'below_1_16', '1_16_to_1_8', '1_8_to_1_4', '1_4_to_1_2', 'above_1_2')
RIDGE = 1e-3


def band_map(height, width):
    if min(height, width) < 16:
        raise ValueError('image dimensions must be >=16')
    frequency = np.maximum(np.arange(height)[:, None] / height,
                           np.arange(width)[None, :] / width)
    bands = 1 + np.searchsorted([1/16, 1/8, 1/4, 1/2], frequency, side='right')
    bands[0, 0] = 0
    return bands


def image(value):
    a = np.asarray(value)
    if a.ndim != 3 or a.shape[-1] != 3 or a.dtype != np.uint8:
        raise ValueError('uint8 HWC RGB required')
    return a.astype(np.float64) / 255


def coefficients(value):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (6,) or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError('six finite coefficients in [0,1] required')
    return a


def statistics(lq, raw, gt):
    """Training-only sufficient statistics, equally weighting each fixed-size crop."""
    x, y, target = image(lq), image(raw), image(gt)
    if x.shape != y.shape or x.shape != target.shape:
        raise ValueError('unaligned training triple')
    delta = dctn(y-x, axes=(0, 1), norm='ortho')
    desired = dctn(target-x, axes=(0, 1), norm='ortho')
    bands = band_map(*x.shape[:2])
    return {'energy': [float(np.square(delta[bands == k]).sum() / x.size) for k in range(6)],
            'cross': [float((delta[bands == k]*desired[bands == k]).sum() / x.size) for k in range(6)]}


def fit(stats, parent_alpha):
    """One analytic fit, no grid search, validation feedback or per-image coefficients."""
    if not stats or not 0 <= parent_alpha <= 1:
        raise ValueError('training statistics and parent alpha required')
    energy = np.asarray([s['energy'] for s in stats], dtype=np.float64)
    cross = np.asarray([s['cross'] for s in stats], dtype=np.float64)
    if energy.shape != cross.shape or energy.shape[1:] != (6,) or not np.isfinite(energy).all() or not np.isfinite(cross).all() or (energy < 0).any():
        raise ValueError('invalid sufficient statistics')
    energy, cross = energy.mean(0), cross.mean(0)
    penalty = RIDGE * np.maximum(energy, 1e-12)
    return coefficients(np.clip((cross + penalty*parent_alpha)/(energy+penalty), 0, 1))


def apply(lq, raw, weights):
    """Reflecting DCT boundary; fixed global task weights, no GT/path/category input."""
    x, y, weights = image(lq), image(raw), coefficients(weights)
    if x.shape != y.shape:
        raise ValueError('unaligned LQ and FLUX')
    bands = band_map(*x.shape[:2])
    residual = dctn(y-x, axes=(0, 1), norm='ortho')
    restored = x + idctn(residual*weights[bands, None], axes=(0, 1), norm='ortho')
    # One explicit quantization after the already-quantized frozen FLUX output.
    return np.rint(np.clip(restored, 0, 1)*255).astype(np.uint8)


def assigned(items, node, local, nodes):
    if len(set(nodes)) != len(nodes) or not nodes or any(n not in range(3) for n in nodes) or node not in nodes or local not in (0, 1):
        raise ValueError('unique registered nodes and two local GPUs required')
    rank = nodes.index(node)*2 + local
    return rank, items[rank::2*len(nodes)]
