import numpy as np

def _finger_up(lm: np.ndarray, tip: int, pip: int) -> bool:
    # MediaPipe normalized: y yukarı doğru küçülür
    return lm[tip, 1] < lm[pip, 1]

def _count_fingers(lm: np.ndarray) -> int:
    """Kaç parmak açık olduğunu sayar"""
    fingers = [
        _finger_up(lm, 8, 6),   # işaret
        _finger_up(lm, 12, 10), # orta
        _finger_up(lm, 16, 14), # yüzük
        _finger_up(lm, 20, 18), # serçe
    ]
    thumb_out = abs(lm[4, 0] - lm[3, 0]) > 0.05  # başparmak

    count = sum(fingers)
    if thumb_out:
        count += 1

    return count

def classify_rules(lm: np.ndarray) -> str:
    idx_up  = _finger_up(lm, 8, 6)
    mid_up  = _finger_up(lm, 12, 10)
    ring_up = _finger_up(lm, 16, 14)
    pink_up = _finger_up(lm, 20, 18)
    thumb_out = abs(lm[4, 0] - lm[3, 0]) > 0.05

    # Özel hareketler (öncelikli)
    if idx_up and mid_up and ring_up and pink_up and thumb_out:
        return "OPEN_PALM"
    if (not idx_up) and (not mid_up) and (not ring_up) and (not pink_up):
        return "FIST"
    if idx_up and (not mid_up) and (not ring_up) and (not pink_up):
        return "POINT"
    if thumb_out and (not idx_up) and (not mid_up) and (not ring_up) and (not pink_up):
        return "THUMBS_UP"

    # Parmak sayısına göre hareketler
    finger_count = _count_fingers(lm)
    if finger_count == 1 and idx_up:
        return "ONE_FINGER"
    elif finger_count == 2 and idx_up and mid_up:
        return "TWO_FINGERS"
    elif finger_count == 3 and idx_up and mid_up and ring_up:
        return "THREE_FINGERS"
    elif finger_count == 4 and idx_up and mid_up and ring_up and pink_up:
        return "FOUR_FINGERS"
    elif finger_count == 5:
        return "FIVE_FINGERS"

    return "UNKNOWN"
