import os
from glob import glob
from typing import List, Any, Iterable
import random
from dataclasses import dataclass, fields

import numpy as np
from PIL import Image


#Sequence lenght?
#Different sequences. Different sizes.

def split_list_complete_chunks(x: list, chunk_size: int) -> List[List[Any]]:
    chunks = [x[i: i+chunk_size] for i in range(0, len(x), chunk_size)]
    if len(chunks[-1]) < chunk_size:
        chunks = chunks[:-1]
    return chunks


def read_pose_txt(path: str) -> List[np.ndarray]:
    mat = np.loadtxt(path)
    poses = [row.reshape(3, 4) for row in mat]
    return poses


def read_times_file(path: str) -> List[int]:
    times = np.loadtxt(path)
    assert len(times.shape) == 1
    return times.tolist()


@dataclass
class View:
    path: str
    pose: np.ndarray
    time_stamp: float
    source: str

    @property
    def image(self) -> Image.Any:
        return Image.open(self.path)

    def __repr__(self) -> str:
        return f"View({self.source}, stamp={self.time_stamp:.2f})"


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

        self.data = [
            View(path, pose, stamp, f"kitti_odom_{self.key}")
            for path, pose, stamp in zip(img_paths, poses, times)
        ]
        self.data.sort(key = lambda x: x.time_stamp)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> View:
        return self.data[i]

    def tolist(self) -> List[View]:
        return list(self.data)


class KittiOdom:
    def __init__(
        self,
        data_root: str,
        seq_len: int=24,
    ):
        self.data_root = data_root
        poses_path = os.path.join(data_root, "poses")
        available_posed_seqs = [
            os.path.splitext(f)[0]
            for f in glob("*.txt", root_dir=poses_path)
        ]

        self.kitty_odom_sequences = []
        for seq in available_posed_seqs:
            try:
                kitti_posed_sequence = KittiOdomSequence(data_root, seq)
                self.kitty_odom_sequences.append(kitti_posed_sequence)
            except Exception as e:
                continue

        self.sequences = self.chunk_sequences(seq_len)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, i: int) -> List[View]:
        return self.sequences[i]

    def chunk_sequences(self, seq_len: int) -> List[List[View]]:
        chunks = []
        for seq in self.kitty_odom_sequences:
            chunks += split_list_complete_chunks(seq.tolist(), seq_len)
        return chunks

    def reset_seq_len(self, seq_len: int) -> None:
        self.sequences = self.chunk_sequences(seq_len)

    def shuffle(self) -> None:
        random.shuffle(self.sequences)

    def get_batches(self, *,batch_size: int, sample_size: int) -> Iterable:
        for batch_i in range(0, self.__len__(), batch_size):
            batch = self.sequences[batch_i: batch_i+batch_size]
            batch = [random.sample(seq, sample_size) for seq in batch]
            attributes = ['image'] + [f.name for f in fields(View)]
            yield {
                att: [
                    [view.__getattribute__(att) for view in seq]
                    for seq in batch
                ]
                for att in attributes
            }
        
