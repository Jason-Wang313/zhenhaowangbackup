"""Fixed classical local alignment of a frozen prediction to an observed image."""
import cv2
import numpy as np

MAX_COMPONENT_PIXELS = 16.0
MATCH_SIGMA_PIXELS = 2.0


def validate_pair(reference, prediction):
    if (reference.dtype != np.uint8 or prediction.dtype != np.uint8
            or reference.shape != prediction.shape or reference.ndim != 3
            or reference.shape[2] != 3 or min(reference.shape[:2]) < 32):
        raise ValueError('aligned uint8 HWC RGB images, minimum32 pixels required')


def warp(prediction, backward_flow):
    if (backward_flow.shape != (*prediction.shape[:2], 2)
            or backward_flow.dtype != np.float32 or not np.isfinite(backward_flow).all()):
        raise ValueError('finite float32 backward sampling flow required')
    yy, xx = np.mgrid[:prediction.shape[0], :prediction.shape[1]].astype(np.float32)
    return cv2.remap(prediction, xx + backward_flow[..., 0], yy + backward_flow[..., 1],
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def align(reference, prediction):
    """No GT or dataset identifiers. Caller defines the observed reference."""
    validate_pair(reference, prediction)
    gray = [cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (0, 0), MATCH_SIGMA_PIXELS)
            for image in (reference, prediction)]
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = estimator.calc(gray[0], gray[1], None)
    if not np.isfinite(flow).all():
        raise ValueError('nonfinite estimated flow')
    clipped = np.clip(flow, -MAX_COMPONENT_PIXELS, MAX_COMPONENT_PIXELS)
    output = warp(prediction, clipped)
    return output, {
        'flow_RMS_pixels': float(np.sqrt(np.mean(flow ** 2))),
        'flow_max_absolute_component_pixels': float(np.abs(flow).max()),
        'component_clip_fraction': float(np.mean(np.abs(flow) > MAX_COMPONENT_PIXELS)),
        'retained_flow_RMS_pixels': float(np.sqrt(np.mean(clipped ** 2))),
    }
