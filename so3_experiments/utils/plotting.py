"""Copyright (c) Dreamfold."""
import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
from PIL import Image

# import orthogonal_group
import io
import plotly.graph_objects as go


def plot_circular_hist(y, bins=40, fig=None):
    """
    :param y: angle values in [0, 1]
    :param bins: number of bins
    :param fig: matplotlib figure object
    """

    theta = np.linspace(0, 2 * np.pi, num=bins, endpoint=False)
    radii = np.histogram(y, bins, range=(0, 2 * np.pi), density=True)[0]

    # # Display width
    width = (2 * np.pi) / (bins * 1.25)

    # Construct ax with polar projection
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    # Set Orientation
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xlim(0, 2 * np.pi)  # workaround for a weird issue
    ax.set_xticks(np.pi / 180.0 * np.linspace(0, 360, 8, endpoint=False))

    # Plot bars:
    _ = ax.bar(x=theta, height=radii, width=width, color="gray")

    # Grid settings
    ax.set_rgrids([])

    return fig, ax


def fig2pil(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.tostring_rgb()
    pil_im = Image.frombytes("RGB", (w, h), buf)
    plt.close("all")
    return pil_im


def plot_scatter3D(xyz, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-1.0, 1.0)):
    fig = plt.figure(figsize=(8, 8), dpi=80)
    ax = plt.axes(projection="3d")
    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.5)
    ax.axes.set_xlim3d(xlim[0], xlim[1])
    ax.axes.set_ylim3d(ylim[0], ylim[1])
    ax.axes.set_zlim3d(zlim[0], zlim[1])
    return fig


def eulerAnglesToRotationMatrix(theta):
    """https://learnopencv.com/rotation-matrix-to-euler-angles/"""

    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def isRotationMatrix(R):
    """https://learnopencv.com/rotation-matrix-to-euler-angles/"""
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    """https://learnopencv.com/rotation-matrix-to-euler-angles/"""

    # assert(isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def plot_so3(x):
    return plot_scatter3D(
        np.array([rotationMatrixToEulerAngles(x[i]) for i in range(len(x))]),
        (-math.pi, math.pi),
        (-math.pi / 2, math.pi / 2),
        (-math.pi, math.pi),
    )


def log_multimodal(x):
    """Multimodal distribution on O(n) with components centered at the identity
    matrix and the pure reflection.
    """
    Id = np.eye(3)
    Ra = np.diag(np.array([-1.0, -1.0, 1]))
    Rb = np.diag(np.array([-1.0, 1.0, -1]))
    scale = 0.5
    lp = 0.0
    lp += np.exp(-0.5 * np.square(x - Id).sum((-1, -2)) / np.square(scale))
    lp += np.exp(-0.5 * np.square(x - Ra).sum((-1, -2)) / np.square(scale))
    lp += np.exp(-0.5 * np.square(x - Rb).sum((-1, -2)) / np.square(scale))
    return np.log(lp)


def plot_so3_multimodal_density(log_p_fn=log_multimodal):
    npoints = 40j
    X, Y, Z = np.mgrid[-3.14:3.14:npoints, -1.57:1.57:npoints, -3.14:3.14:npoints]
    points = np.concatenate(
        [
            eulerAnglesToRotationMatrix([x, y, z])[None]
            for x, y, z in zip(X.flatten(), Y.flatten(), Z.flatten())
        ],
        0,
    )
    values = np.exp(log_p_fn(points)).flatten()

    vol = 4 * np.pi**2 / abs(npoints) ** 3
    values /= values.sum() * vol

    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values.flatten(),
            isomin=0.055,
            isomax=0.12,
            opacity=0.1,
            surface_count=20,
        )
    )
    fig.show()

    buf = io.BytesIO()
    print("writing image...")
    fig.write_image(buf, width=800, height=600, scale=2)
    buf.seek(0)
    print("done")

    return Image.open(buf)
