#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
from typing import NamedTuple

import numpy as np
import torch


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def geom_transform_points(points, transf_matrix):
    p, _ = points.shape
    ones = torch.ones(p, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    rt = np.zeros((4, 4))
    rt[:3, :3] = R.transpose()
    rt[:3, 3] = t
    rt[3, 3] = 1.0
    return np.float32(rt)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    rt = np.zeros((4, 4))
    rt[:3, :3] = R.transpose()
    rt[:3, 3] = t
    rt[3, 3] = 1.0

    c2w = np.linalg.inv(rt)
    cam_center = c2w[:3, 3]
    cam_center = (cam_center + translate) * scale
    c2w[:3, 3] = cam_center
    rt = np.linalg.inv(c2w)
    return np.float32(rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    p = torch.zeros(4, 4)

    z_sign = 1.0

    p[0, 0] = 2.0 * znear / (right - left)
    p[1, 1] = 2.0 * znear / (top - bottom)
    p[0, 2] = (right + left) / (right - left)
    p[1, 2] = (top + bottom) / (top - bottom)
    p[3, 2] = z_sign
    p[2, 2] = z_sign * zfar / (zfar - znear)
    p[2, 3] = -(zfar * znear) / (zfar - znear)
    return p


def getProjectionMatrix_with_principal(znear, zfar, fx, fy, cx, cy, width, height):
    """
    Build an off-center projection matrix from camera intrinsics.

    Assumptions:
    - Pixel coordinates use u to the right and v downward.
    - The principal point (cx, cy) is defined with the top-left pixel as the origin.
    """

    # X direction: coordinates on the near plane corresponding to u=0 and u=width.
    left = (cx - width) * znear / fx
    right = cx * znear / fx

    # Y direction: v=0 is at the top of the image and v=height is at the bottom.
    top = cy * znear / fy
    bottom = (cy - height) * znear / fy

    p = torch.zeros(4, 4, dtype=torch.float32)

    z_sign = 1.0

    p[0, 0] = 2.0 * znear / (right - left)
    p[1, 1] = 2.0 * znear / (top - bottom)
    p[0, 2] = (right + left) / (right - left)
    p[1, 2] = (top + bottom) / (top - bottom)
    p[3, 2] = z_sign
    p[2, 2] = z_sign * zfar / (zfar - znear)
    p[2, 3] = -(zfar * znear) / (zfar - znear)

    return p


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))
