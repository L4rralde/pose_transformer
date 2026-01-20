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
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

image_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])


def pose_seq_transform(pose_seq: List[np.ndarray]) -> List[np.ndarray]:
    """
    Receives a list of poses (they can be either 3x4 or 4x4 matrices)
    Sets the first pose as origin and scales the trajectory such as each pose lies
    within a unit sphere.
    Poses are transformed as follows:
        C' = S C S_s^{-1}
    Where s = 1/max_i(||t_i - t_0||)
    And I = S C_0 S_s^{-1} \implies S = S_s C_0^{-1}
    TWe use the close form of the operation.
    """
    pose_seq_h = [dutils.to_homogenoeus(pose) for pose in pose_seq]
    if len(pose_seq_h) == 1:
        scale = 1.0
    else:
        max_distance = dutils.max_distance_from_first(pose_seq_h)
        if max_distance == 0:
            raise RuntimeError("All poses share the same position?")
        scale = 1.0/max_distance
    
    #First, ensure evry matix is a homogeneus matrix
    c0_inv = dutils.closed_form_se3_inv(pose_seq_h[0])
    t_pose_seq_h = []
    for pose in pose_seq_h: #S_s C_0^{-1} C S_s^{-1}
        t_pose = c0_inv @ pose # C_0^{-1} C
        t_pose[:3, 3] *= scale  # S_s _ S_s^{-1}
        t_pose_seq_h.append(t_pose)

    #Additional sanity check
    if not np.allclose(t_pose_seq_h[0], np.identity(4), atol=1e-6):
        print(t_pose_seq_h[0])
        raise RuntimeError("First transformed pose is not the origin")

    max_dist = dutils.max_distance_from_first(t_pose_seq_h)
    if abs(max_dist - 1.0) > 1e-6:
        print(max_dist)
        print(t_pose_seq_h)
        raise RuntimeError("Trajectorie does not lie in the unit sphere")
    
    return t_pose_seq_h


def main():
    model = PoseTransformer(input_size=img_size).to(device)
    optim = torch.optim.AdamW(model.parameters())

    q_loss = nn.MSELoss(reduction='sum')
    t_loss = nn.HuberLoss(reduction='sum')

    #for epoch in range(1, 10 +1):
    kitti_sequencer = KittiOdom('data/kitti-odometry/', 24)
    kitti_sequencer.shuffle()

    epochs = 20

    for epoch in tqdm(range(1, epochs+1)):
        sample_size = ((epoch - 1) % 7) + 2
        batch_size = 128//sample_size
        tqdm.write(f"Using batch size: {batch_size} and sequence length: {sample_size}")
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
            x = x.to(device)

            #2. Data pre-processing
            #Pre-process extrinsics: Set first pose as origin and normalize translations.
            pose = np.asarray([
                pose_seq_transform(pose_seq)
                for pose_seq in batch['pose']
            ])

            #Get translation Tensor.
            t = torch.Tensor(pose[..., :3, 3])
            t = t.to(device)
            #Convert roation matrices to quaternions.
            qvec = torch.Tensor(np.asarray([
                dutils.mat_to_quat(rot_seq)
                for rot_seq in pose[..., :3, :3]
            ]))
            qvec = qvec.to(device)

            #3. Forward pass, 
            optim.zero_grad()
            preds = model(x)
            loss = q_loss(preds['qvec'], qvec) + t_loss(preds['t'], t)

            #4. Backward pass,
            loss.backward()

            #5. Model update
            optim.step()

            numeric_loss += loss.data.cpu().item()
    
        tqdm.write(f"{epoch}: {numeric_loss}")


if __name__ == '__main__':
    main()
