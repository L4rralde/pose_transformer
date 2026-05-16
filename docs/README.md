# PoseFormer: Camera Pose Transformer

By the moment this is limited to predict only camera extrinsics, but in the future camera intrinsics may be predicted as well.

Let $\{I\}$ be a sequence of images, we want to approximate a function $f(\cdot)$ to estimate the sequence of the poses of the camera $\{[R | t]\}$ corresponding to each input image. The expression is as below:

$$
    \{[R | t]\} = f(\{I\})
$$

For simplicity, let $\mathcal X = [R|t]$ denote the pose matrix. 
Let $\tilde{\mathcal X}$ denote its homogeneous form.

Following trends in 3D geometry foundation models, $f$ is desinged to be a concatenation of the following form:

1. Per-image encoding
2. Per-sequence encoding
3. Projection head

I'll experiment using EUPE as image encoder. 
The per-sequence encoding wuold be a trainable alternate
attention block composed of ViT blocks.
The projection head could be as simply as a MLP.

### Per-Image encoder

I'll experiment using EUPE as image encoder. 
I guess this returns a vector of patch tokens and the CLS tokens
for each image of the input sequence.


### Per-sequence encoding

We drop the global CLS token and replace it with a trainable
CAM token. Actually, there are 2 CAM tokens. One is specifically
for the first image while the second is shared to all other images.

TODO. Do we need a positional embedding?


### Projection head

For the first version, I'll use a MLP to transform each 
final CAM token into a displacement $t$ vector and a
orientation $q$ quaternion. To normalize outputs, the first
pose will be the identity while the displacement between the 
first and the mth poses is unit-length.

## Ground-truth preprocessing

Let $\{I\}$ an image sequence. $\{{ \mathcal X_{gt} = [R_{gt}|t_{gt}]} \}$ the sequence of ground truth poses.

### Are poses cam-to-world or world-to-cam?

#### If they are cam-to-world

Left-multiply by the inverse of the first pose.
Since now all coordinates are wrt the first camera.

Let me explain with this example.

$\mathcal X_i \tilde 0$ is the origin of camera 
$i$ in hypothetical world-coordinates. 
$\mathcal X_1^{-1}$ just projects $\mathcal X_i \tilde 0$ 
wrt camera 0, i.e., it computes the position 
in $\mathcal X_1$ coordinates.

In summary, we transform all poses to:
$$
    \mathcal X_1^{-1} \mathcal X_i
$$


#### If they are world-to-cam

Just invert them. $\mathcal X_i^{-1}$ is cam-to-world. 
$\mathcal X_1 \mathcal X_i^{-1}$ is cam-to-world and is
equivalent to the aformentioned deduced expression.
Hence, the world-to-cam equivalence is:

$$
    \mathcal X_i \mathcal X_1^{-1}
$$

### Normalizing the magnitude.

$$
    \begin{bmatrix}
        sI & 0 \\
        0 & 1
    \end{bmatrix}
    \begin{bmatrix}
        R & t \\
        0 & 1
    \end{bmatrix}
    \begin{bmatrix}
        s^{-1}I & 0 \\
        0 & 1
    \end{bmatrix}
    =
    \begin{bmatrix}
        sR & st \\
        0 & 1
    \end{bmatrix}
    \begin{bmatrix}
        s^{-1}I & 0 \\
        0 & 1
    \end{bmatrix}
    =
    \begin{bmatrix}
        R & st \\
        0 & 1
    \end{bmatrix}
$$

We just need to normalize the trajectory. Somehow.

possible options:

1. The whole trajectory distance is unitary. This expects **cam-to-world** transformations.

$$
    \sum_{i=2}^n ||t_{i} - t_{i-1}|| = 1
$$

2. the trajectory lies in a unit sphere:

$$
    \max ||t_i|| = 1
$$

3. The distance between the first and the second camera is unitary:

$$
    ||t_2 - t_1|| = 1
$$

This is unstable since, possibly,  $t_2 \approx t_1$

Here, we use the second. But need to first check 
if poses are world-to-cam or cam-to-world


