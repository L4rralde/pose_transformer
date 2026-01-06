from typing import List

from PIL import Image
import numpy as np


def load_image(path: str) -> Image.Any:
    return Image.open(path)


class SquareResize:
    def __init__(self, size) -> None:
        self.size = size

    def __call__(self, img: Image.Any) -> Image.Any:
        return img.resize((self.size, self.size))


class ResizeForTokenization:
    def __init__(self, max_size: int, patch_size: int) -> None:
        if max_size % patch_size != 0:
            raise ValueError("Max_size must be divisible by patch size")
        self.max_size = max_size
        self.patch_size = patch_size

    def __call__(self, img: Image.Any) -> Image.Any:
        w, h = img.size
        scale = float(self.max_size)/w if w>h else float(self.max_size)/h
        w = int(scale * w)
        h = int(scale * h)
        w = ResizeForTokenization.find_nearest_multiple(w, self.patch_size)
        h = ResizeForTokenization.find_nearest_multiple(h, self.patch_size)

        return img.resize((w, h))

    @staticmethod
    def find_nearest_multiple(x: int, divisor: int) -> int:
        if x < divisor:
            raise ValueError("Input must be greater than divisor")
        if x % divisor == 0:
            return x
        low = divisor * (x//divisor)
        high = low + divisor
        if abs(x - low) < abs(x - high):
            return low
        return high


def set_first_pose_as_origin(poses: List[np.ndarray]) -> List[np.ndarray]:
    r0 = poses[0][:3, :3]
    t0 = poses[0][:3, 3]
    new_poses = []
    for p in poses:
        new_p = np.empty_like(p)
        new_p[:3, :3] = r0.T @ p[:3, :3]
        new_p[:3, 3] = r0.T @ (p[:3, 3] - t0)
        new_poses.append(new_p)

    return new_poses


def max_distance_normalization(poses: List[np.ndarray]) -> List[np.ndarray]:
    positions = [p[:3, 3] for p in poses]
    distances = [np.linalg.norm(pos) for pos in positions]
    scale = max(max(distances), 1.0)
    new_poses = [p.copy() for p in poses]
    for p in new_poses:
        p[:3, :] /=  scale
    
    return new_poses


#FUTURE: This code is vibe code. Replace with yours later.
def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion (w, x, y, z).

    Assumes R is a valid rotation matrix.
    """
    assert R.shape == (3, 3)

    q = np.empty(4, dtype=np.float64)
    trace = np.trace(R)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s

    # Normalize for safety
    q /= np.linalg.norm(q)
    return q
