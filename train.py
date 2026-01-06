from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm

from poseformer.transformer import PoseTransformer
from poseformer.training.datasets.kitti_odom import KittiOdom
import poseformer.data_utils as dutils


img_size = (240, 320)


image_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])

def pose_seq_transform(pose_seq: List[np.ndarray]) -> List[np.ndarray]:
    return dutils.max_distance_normalization(
        dutils.set_first_pose_as_origin(pose_seq)
    )


def main():
    model = PoseTransformer(input_size=img_size)
    optim = torch.optim.AdamW(model.parameters())

    q_loss = nn.MSELoss(reduction='sum')
    t_loss = nn.HuberLoss(reduction='sum')

    #for epoch in range(1, 10 +1):
    kitti_sequencer = KittiOdom('data/kitti-odometry/', 24)
    kitti_sequencer.shuffle()

    epochs = 20

    for epoch in tqdm(range(1, epochs+1)):
        batch_size = 16
        sample_size = (epoch%8) + 1
        numeric_loss = 0.0
        for batch in kitti_sequencer.get_batches(batch_size=batch_size, sample_size=sample_size):
            #1. Sanity Check
            #Check if all views of a sequence come from same source
            for source_seqs in batch['source']:
                if len(set(source_seqs)) != 1:
                    raise RuntimeError("Found views from differents sources in sequence")
            #TODO: Check that the time delta in a sequence is small enough
            x = torch.stack([
                torch.stack([
                    image_transform(view)
                    for view in view_seq
                ])
                for view_seq in batch['image']
            ])

            #2. Data pre-processing
            #Pre-process extrinsics: Set first pose as origin and normalize translations.
            pose = np.asarray([
                pose_seq_transform(pose_seq)
                for pose_seq in batch['pose']
            ])

            #Get translation Tensor.
            t = torch.Tensor(pose[..., :3, 3])
            #Convert roation matrices to quaternions.
            qvec = torch.Tensor(np.asarray([
                dutils.mat_to_quat(rot_seq)
                for rot_seq in pose[..., :3, :3]
            ]))

            #3. Forward pass, 
            optim.zero_grad()
            preds = model(x)
            loss = q_loss(preds['qvec'], qvec) + t_loss(preds['t'], t)

            #4. Backward pass,
            loss.backward()

            #5. Model update
            optim.step()

            numeric_loss += loss.data.item()
    
        tqdm.write(f"{epoch}: {numeric_loss}")


if __name__ == '__main__':
    main()
