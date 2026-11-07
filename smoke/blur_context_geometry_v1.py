"""Fixed coordinates and pixel statistics for a read-only training-image diagnosis."""


def crop(array, item, margin=0):
    x, y = item['crop_xy']
    h, w = array.shape[:2]
    if (array.ndim != 3 or array.shape[2] != 3 or type(margin) is not int
            or not 0 <= margin < 256 or not 0 <= x <= w - 512 or not 0 <= y <= h - 512):
        raise ValueError('invalid fixed512 crop geometry')
    return array[y + margin:y + 512 - margin, x + margin:x + 512 - margin].copy()


def pixel_statistics(lq, gt, full, standalone, item):
    import numpy as np
    if lq.shape != gt.shape or gt.shape != full.shape or standalone.shape != (512, 512, 3):
        raise ValueError('mismatched images')
    if any(a.dtype != np.uint8 for a in (lq, gt, full, standalone)):
        raise ValueError('fixed uint8 images required')
    error = (full.astype(np.float64) - gt.astype(np.float64)) / 255
    lq_error = (lq.astype(np.float64) - gt.astype(np.float64)) / 255
    outside = np.ones(lq.shape[:2], dtype=bool)
    x, y = item['crop_xy']
    from_full = crop(full, item)
    outside[y:y + 512, x:x + 512] = False
    if not outside.any():
        raise ValueError('nonempty outside region required')
    difference = (from_full.astype(np.float64) - standalone.astype(np.float64)) / 255
    return {'outside_trained_crop_MSE': float(np.mean(error[outside] ** 2)),
            'outside_trained_crop_LQ_MSE': float(np.mean(lq_error[outside] ** 2)),
            'same_crop_output_difference_RMS_RGB01': float(np.sqrt(np.mean(difference ** 2))),
            'center384_output_difference_RMS_RGB01': float(np.sqrt(np.mean(difference[64:-64, 64:-64] ** 2))),
            'same_crop_output_difference_MAE_RGB01': float(np.mean(np.abs(difference)))}
