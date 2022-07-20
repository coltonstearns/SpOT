import numpy as np
import open3d as o3d
from pytorch3d.transforms import RotateAxisAngle

import torch

AXIS_MAP = {'x' : np.array([1.0, 0.0, 0.0]), 
            'y' : np.array([0.0, 1.0, 0.0]),
            'z' : np.array([0.0, 0.0, 1.0])}


def boxpoints2canonical(points, boxes, out_refframe=None):
    """

    Args:
        points (torch.tensor): B x T x N x 3 tensor of object points
        boxes: (torch.tensor): B x T x 7 tensor of object boxes
        out_refframe (torch.tensor): size B x 7 of bounding box output reference frame

    Returns:
        canonical_points (torch.tensor): B x T x N x 3 tensor of points in non-scaled aligned reference frame
    """
    B, T, N, _ = points.size()
    canon_pnts = points[:, :, :, :3].clone()
    canon_pnts -= boxes[:, :, :3].view(B, T, 1, 3)
    canon_pnts = canon_pnts.view(B*T, N, 3)

    # per-frame rotations
    # todo: figure out why our yaw is negative for this!
    batch_rot = RotateAxisAngle(-boxes[:, :, 6].flatten(), axis="Z", degrees=False).to(boxes.device)
    canon_pnts = batch_rot.transform_points(canon_pnts).view(B, T, N, 3)

    # if applicable, rotate into target coordinate frame
    if out_refframe is not None:
        rot_mat = RotateAxisAngle(-out_refframe[:, 6].repeat_interleave(T), axis="Z", degrees=False).to(boxes.device)
        rot_mat_inv = rot_mat.inverse()
        canon_pnts = rot_mat_inv.transform_points(canon_pnts.view(B*T, N, 3)).view(B, T, N, 3)
        canon_pnts += out_refframe[:, :3].view(B, 1, 1, 3)

    return canon_pnts


def lidar2bev_box(boxes, bev_offset):
    bev_boxes = torch.cat([boxes[:, :2].clone() - bev_offset, boxes[:, :2].clone() - bev_offset, -boxes[:, 6:7].clone()], dim=1)
    w_l = boxes[:, 3:5]
    bev_boxes[:, 0:2] -= w_l
    bev_boxes[:, 2:4] += w_l
    return bev_boxes


def rotmats2yaws(rotmats):
    """
    Converts rotation matrices to yaws.
    Args:
        rotmats (torch.tensor): tensor size (B, 3, 3)

    Returns:
        yaws (torch.tensor): tensor size (B,)
    """
    v = rotmats @ torch.tensor([1, 0, 0]).double().view(3, 1).to(rotmats)  # outputs B x T x 3 x 1
    v = v.view(-1, 3)
    yaws = torch.atan2(v[:, 1], v[:, 0])
    return yaws


def translation2affine(t):
    affine_matrix = torch.eye(4).to(t).unsqueeze(0).repeat(t.size(0), 1, 1)
    affine_matrix[:, :3, 3] = t
    return affine_matrix



def procrustes(from_points, to_points):
    """
    Implementation of the Procrustes step
    Args:
        from_points (np.array): (4,3)
        to_points (np.array): (4,3)
    Returns: Scale (float), Rotation (3,3), Translation (3)
    """
    assert len(from_points.shape) == 2, \
        "from_points must be a m x n array"
    assert from_points.shape == to_points.shape, \
        "from_points and to_points must have the same shape"

    N, m = from_points.shape

    mean_from = from_points.mean(axis=0)
    mean_to = to_points.mean(axis=0)

    delta_from = from_points - mean_from  # N x m
    delta_to = to_points - mean_to  # N x m

    sigma_from = (delta_from * delta_from).sum(axis=1).mean()
    sigma_to = (delta_to * delta_to).sum(axis=1).mean()

    cov_matrix = delta_to.T.dot(delta_from) / N  # Not really a covariance...

    try:
        U, d, V_t = np.linalg.svd(cov_matrix, full_matrices=True)
    except np.linalg.LinAlgError as e:
        print(e)
        print("Linalg error in Procrustes.")
        return None
    cov_rank = np.linalg.matrix_rank(cov_matrix)
    S = np.eye(m)

    if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
        S[m - 1, m - 1] = -1
    elif cov_rank < m - 1:
        # print("Not full rank!")
        return None
        # raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))

    R = U.dot(S).dot(V_t)
    c = (d * S.diagonal()).sum() / sigma_from
    t = mean_to - c * R.dot(mean_from)
    return c, R, t


def kabsch(canonical_points, predicted_points):
    """
    Implementation of the Kabsch step
    Args:
        canonical_points (np.array): (4,3)
        predicted_points (np.array): (4,3)
    Returns: Rotation (3,3) and translation (3)
    """
    canonical_mean = np.mean(canonical_points, axis=0)
    predicted_mean = np.mean(predicted_points, axis=0)

    canonical_centered = canonical_points - np.expand_dims(canonical_mean, axis=0)
    predicted_centered = predicted_points - np.expand_dims(predicted_mean, axis=0)

    cross_correlation = predicted_centered.T @ canonical_centered

    u, s, vt = np.linalg.svd(cross_correlation)

    rotation = u @ vt

    det = np.linalg.det(rotation)

    if det < 0.0:
        vt[-1, :] *= -1.0
        rotation = np.dot(u, vt)

    translation = predicted_mean - canonical_mean
    translation = np.dot(rotation, translation) - np.dot(rotation, predicted_mean) + predicted_mean

    return rotation, translation

def weighted_procrustes(X, Y, w,  eps=np.finfo(np.float32).eps):
  """
  X: torch tensor N x 3; NOCS
  Y: torch tensor N x 3; Input
  w: torch tensor N x 1
  """
  # https://ieeexplore.ieee.org/document/88573
  assert len(X) == len(Y)
  W1 = torch.abs(w).sum()
  w_norm = w / (W1 + eps)
  mux = (w_norm * X).sum(0, keepdim=True)
  muy = (w_norm * Y).sum(0, keepdim=True)  # torch geometric can get these quantities via mean-pool

  # Use CPU for small arrays
  Sxy = (Y - muy).t().mm(w_norm * (X - mux)).cpu().double()  # mean-centering is easy via indexing. batchwise-matrix-mul
  # so we end up with a batch of arrays, (3 x N) @ (N x 3); can I zero-pad? yes, and it is even equivalent! --> use pytorch3d!

  U, D, V = Sxy.svd()
  S = torch.eye(3).double()
  if U.det() * V.det() < 0:
    S[-1, -1] = -1

  R = U.mm(S.mm(V.t())).float()
  t = (muy.cpu().squeeze() - R.mm(mux.cpu().t()).squeeze()).float()

  # compute yaw from rotmat
  v = R @ torch.tensor([1, 0, 0]).float().view(3, 1)  # outputs 3 x 1
  yaw = torch.atan2(v[1], v[0])  # size (1,)

  return R.to(X), t.to(X), yaw.to(X)

def batch_weighted_procrustes(X, Y, w,  eps=np.finfo(np.float32).eps):
  """
  X: torch tensor B x N x 3; NOCS
  Y: torch tensor B x N x 3; Input
  w: torch tensor B x N x 1
  """
  # https://ieeexplore.ieee.org/document/88573
  assert len(X) == len(Y)
  B = X.size(0)
  W1 = torch.abs(w).sum(dim=1, keepdim=True)
  w_norm = w / (W1 + eps)
  mux = (w_norm * X).sum(dim=1, keepdim=True)
  muy = (w_norm * Y).sum(dim=1, keepdim=True)  # torch geometric can get these quantities via mean-pool

  # Use CPU for small arrays
  Sxy = (Y.cpu() - muy.cpu()).transpose(1, 2) @ (w_norm.cpu() * (X.cpu() - mux.cpu()))
  Sxy = Sxy.double()
  Sxy += (torch.rand(3, 3) * eps).double()  # for numerical stability
  # mean-centering is easy via indexing. batchwise-matrix-mul

  U, D, V = torch.linalg.svd(Sxy)
  S = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(Sxy)
  inv_det = torch.where((U.det() * V.det()) < 0, -torch.ones(B).to(U), torch.ones(B).to(Sxy))
  S[:, -1, -1] = inv_det

  R = U @ S @ V.transpose(1, 2)
  R = R.float().to(X.device)
  t = (muy.squeeze() - (R @ mux.transpose(1, 2)).squeeze()).float().to(X.device)

  return R, t