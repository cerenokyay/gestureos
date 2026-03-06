import numpy as np

def _finger_up(lm: np.ndarray, tip: int, pip: int) -> bool:
    # MediaPipe normalized: y yukarı doğru küçülür
    return lm[tip, 1] < lm[pip, 1]

def classify_rules(lm: np.ndarray) -> str:
    idx_up  = _finger_up(lm, 8, 6)
    mid_up  = _finger_up(lm, 12, 10)
    ring_up = _finger_up(lm, 16, 14)
    pink_up = _finger_up(lm, 20, 18)

    thumb_out = abs(lm[4, 0] - lm[3, 0]) > 0.05

    if idx_up and mid_up and ring_up and pink_up and thumb_out:
        return "OPEN_PALM"
    if (not idx_up) and (not mid_up) and (not ring_up) and (not pink_up):
        return "FIST"
    if idx_up and (not mid_up) and (not ring_up) and (not pink_up):
        return "POINT"
    if thumb_out and (not idx_up) and (not mid_up) and (not ring_up) and (not pink_up):
        return "THUMBS_UP"

    return "UNKNOWN"