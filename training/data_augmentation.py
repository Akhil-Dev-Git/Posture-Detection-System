"""
Data Augmentation for pose keypoint sequences.
Applies horizontal flip, noise, and temporal jittering to augment training data.
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Key points to mirror horizontally
# nose(0), shoulders(11,12), elbows(13,14), wrists(15,16), hips(23,24), etc.
MIRROR_MAP = {
    1: 3, 3: 1,   # left/right eyes
    4: 6, 6: 4,   # right/left eyes
    2: 2,         # left eye (self-mirror)
    5: 5, 7: 7,   # ears
    0: 0,         # nose
    17: 19, 19: 17, 8: 8,  # left/right ears
    9: 10, 10: 9,  # mouth left/right
    11: 12, 12: 11,   # shoulders
    13: 14, 14: 13,   # elbows
    15: 16, 18: 20,  # wrists, pinks
    17: 18, 18: 17,   # pinks
    19: 20, 20: 19,   # index
    21: 22, 22: 21,   # thumbs
    23: 24, 24: 23,   # hips
    25: 26, 26: 25,   # knees
    27: 28, 28: 27,   # ankles
    29: 30, 30: 29,   # heels
    31: 32, 32: 31,   # foot index
}


def horizontal_flip(kp_array):
    """Mirror keypoints about vertical axis (x -> 1 - x)."""
    flipped = kp_array.copy()
    for a, b in MIRROR_MAP.items():
        if a < 33 and b < 33:
            flipped[b] = kp_array[a].copy()
            flipped[b, 0] = 1 - kp_array[a, 0]  # flip x
    return flipped


def add_noise(kp_array, std=0.01):
    """Add Gaussian noise to keypoint coordinates."""
    noisy = kp_array.copy()
    noisy[:, :3] += np.random.normal(0, std, size=(len(kp_array), 3))
    return np.clip(noisy, 0, 1)


def temporal_jitter(kp_array, max_shift=2):
    """Randomly shift frames forward/backward (temporal augmentation)."""
    augmented = kp_array.copy()
    for _ in range(max_shift):
        shift = np.random.randint(-max_shift, max_shift)
        if abs(shift) <= max_shift and shift != 0:
            augmented = np.roll(augmented, shift, axis=0)
    return augmented


def augment_directory(data_dir, out_dir=None, num_aug=3):
    """Augment all .npy sequences in a directory.

    Args:
        data_dir: directory with .npy sequences
        out_dir: output dir (defaults to data_dir_augmented/)
        num_aug: number of augmented versions per sequence
    """
    if out_dir is None:
        out_dir = data_dir + "_augmented"
    os.makedirs(out_dir, exist_ok=True)

    for npy_file in Path(data_dir).glob("*.npy"):
        seq = np.load(npy_file)
        # Original
        np.save(os.path.join(out_dir, f"orig_{npy_file.name}"), seq)
        # Augmented versions
        for i in range(num_aug):
            aug = seq.copy()
            aug = horizontal_flip(aug)
            aug = add_noise(aug)
            aug = temporal_jitter(aug)
            np.save(os.path.join(out_dir, f"aug{i}_{npy_file.name}"), aug)
    print(f"Augmented {len(list(Path(data_dir).glob('*.npy')))} sequences -> "
          f"{len(list(Path(out_dir).glob('*.npy')))} total")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True,
                        help="Directory with .npy sequences")
    parser.add_argument("--num", type=int, default=3,
                        help="Augmented copies per sequence")
    args = parser.parse_args()
    augment_directory(args.dir, num_aug=args.num)
