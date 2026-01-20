from typing import List

import numpy as np


class Sim3Transform:
    def __init__(
        self,
        s: float=1.0,
        R: np.ndarray=np.eye(3),
        t: np.ndarray=np.zeros(3)
    ) -> None:
        self.s = s
        self.R = R.copy()
        self.t = t.copy()

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> "Sim3Transform":
        sR = matrix[:3, :3]
        s = np.linalg.det(sR)
        R = sR/s
        t = matrix[:3, 3]
        return Sim3Transform(s, R, t)

    def to_list(self) -> List[float, np.ndarray, np.ndarray]:
        return [self.s, self.R.copy(), self.t.copy()]

    def to_short_matrix(self) -> np.ndarray:
        return np.hstack((self.s * self.R, self.t))

    def to_homogeneus(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3] = self.to_short_matrix()
        return matrix

    def copy(self, other: "Sim3Transform") -> "Sim3Transform":
        return Sim3Transform(other.s, other.R, other.t)

    def inverse(self) -> "Sim3Transform":
        raise NotImplementedError("TODO")
