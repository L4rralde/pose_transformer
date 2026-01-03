import os
from glob import glob
from typing import Callable, List, Dict, Any

import numpy as np
from torch.utils.data import Dataset

#Sequence lenght?
#Different sequences. Different sizes.


def read_pose_txt(path: str) -> List[np.ndarray]:
    mat = np.loadtxt(path)
    poses = [row.reshape(3, 4) for row in mat]
    return poses


def read_times_file(path: str) -> List[int]:
    times = np.loadtxt(path)
    assert len(times.shape) == 1
    return times.tolist()


def split_list_complete_chunks(x: list, chunk_size: int) -> List[list]:
    chunks = [x[i: i+chunk_size] for i in range(0, len(x), chunk_size)]
    if len(chunks[-1]) < chunk_size:
        chunks = chunks[:-1]
    for chunk in chunks:
        if len(chunk) != chunk_size:
            raise RuntimeError("Incorrect implementation")
    return chunks


class KittiOdomSequence:
    def __init__(self, root_dir: str, seq: int|str) -> None:
        self.root_dir = root_dir
        self.key = f"{seq: 02d}"if isinstance(seq, int) else seq    
        poses_file_path = os.path.join(self.root_dir, "poses", f"{self.key}.txt")
        times_file_path = os.path.join(self.root_dir, "sequences", self.key, "times.txt")
        imgs_dir_path = os.path.join(self.root_dir, "sequences", self.key, "image_2")

        poses = read_pose_txt(poses_file_path)
        times = read_times_file(times_file_path)
        img_paths = sorted(glob("*.png", root_dir=imgs_dir_path))
        assert len(poses) == len(times) == len(img_paths)
        img_paths = [str(os.path.join(imgs_dir_path, path)) for path in img_paths]
        assert len(poses) == len(times) == len(img_paths)

        self.data = list(zip(img_paths, poses, times, [self.key for _ in range(len(poses))]))
        self.data.sort(key = lambda x: x[2])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> tuple:
        return self.data[i]

    def tolist(self) -> List[tuple]:
        return list(self.data)


class KittiOdomDataset(Dataset):
    def __init__(self, data_root: str, seq_len: int=24, img_transform: Callable|None=None):
        self.data_root = data_root
        self.img_transform = img_transform
        self.seq_len = seq_len
        poses_path = os.path.join(data_root, "poses")
        available_posed_seqs = [
            os.path.splitext(f)[0]
            for f in glob("*.txt", root_dir=poses_path)
        ]

        self.data = []
        for seq in available_posed_seqs:
            try:
                kitti_posed_sequence = KittiOdomSequence(data_root, seq)
                chunks = split_list_complete_chunks(kitti_posed_sequence.tolist(), seq_len)
                self.data += chunks
            except Exception as e:
                print(e)
                continue


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> "KittiOdomSequence":
        return self.data[i]
