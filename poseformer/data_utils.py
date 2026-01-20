from typing import List

from PIL import Image
import numpy as np


def load_image(path: str) -> Image.Image:
    return Image.open(path)


class SquareResize:
    def __init__(self, size) -> None:
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.resize((self.size, self.size))


class ResizeForTokenization:
    def __init__(self, max_size: int, patch_size: int) -> None:
        if max_size % patch_size != 0:
            raise ValueError("Max_size must be divisible by patch size")
        self.max_size = max_size
        self.patch_size = patch_size

    def __call__(self, img: Image.Image) -> Image.Image:
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


def max_distance_from_first(poses: List[np.ndarray]) -> float:
    if len(poses) < 2:
        raise ValueError("Expected a list of different views")
    positions = [p[:3, 3] for p in poses]
    distances = [np.linalg.norm(pos - positions[0]) for pos in positions]
    max_distance = max(distances)

    return max_distance


def closed_form_se3_inv(s3_mat: np.ndarray) -> np.ndarray:
    R = s3_mat[:3, :3]
    t = s3_mat[:3, 3]
    inv = s3_mat.copy()
    inv[:3, :3] = np.transpose(R)
    inv[:3, 3] = -np.transpose(R) @ t
    return inv


def to_homogenoeus(mat: np.ndarray):
    m, _ = mat.shape
    if m == 4:
        return mat
    mat_h = np.eye(4, dtype=mat.dtype)
    mat_h[:3] = mat
    return mat_h


def _sqrt_positive_part(x: np.ndarray) -> np.ndarray:
    """
    sqrt(max(x, 0)) with zero gradient at x=0 (for consistency with torch code)
    """
    return np.sqrt(np.maximum(x, 0.0))


def standardize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Ensure the real (scalar) part is non-negative.
    Quaternion format: (..., 4) with scalar-last (x, y, z, w)
    """
    sign = np.where(q[..., 3:4] < 0, -1.0, 1.0)
    return q * sign


def mat_to_quat(
    matrices: list[np.ndarray] | np.ndarray,
) -> np.ndarray:
    """
    Convert rotation matrices to quaternions using a robust, branch-free method.

    Args:
        matrices: list or array of shape (N, 3, 3)

    Returns:
        quaternions: array of shape (N, 4) in (x, y, z, w) order
    """
    R = np.asarray(matrices, dtype=np.float64)

    if R.ndim != 3 or R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected shape (N, 3, 3), got {R.shape}")

    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    # Step 1: Compute candidate magnitudes
    q_abs = _sqrt_positive_part(
        np.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            axis=1,
        )
    )

    # Step 2: Build quaternion candidates (rijk ordering)
    quat_by_rijk = np.stack(
        [
            np.stack([q_abs[:, 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=1),
            np.stack([m21 - m12, q_abs[:, 1] ** 2, m10 + m01, m02 + m20], axis=1),
            np.stack([m02 - m20, m10 + m01, q_abs[:, 2] ** 2, m12 + m21], axis=1),
            np.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[:, 3] ** 2], axis=1),
        ],
        axis=1,
    )  # shape: (N, 4, 4)

    # Step 3: Normalize candidates
    eps = 0.1
    denom = 2.0 * np.maximum(q_abs, eps)[:, :, None]
    quat_candidates = quat_by_rijk / denom

    # Step 4: Pick best-conditioned candidate
    best = np.argmax(q_abs, axis=1)
    quats = quat_candidates[np.arange(len(R)), best]

    # Step 5: Convert from rijk → ijkr (x, y, z, w)
    quats = quats[:, [1, 2, 3, 0]]

    # Step 6: Standardize sign (w ≥ 0)
    quats = standardize_quaternion(quats)

    return quats
