import numpy as np
import open3d as o3d

from anypick_dk.constants import PREGRASP_OFFSET
from pydrake.all import BezierCurve, CompositeTrajectory, PiecewisePolynomial, PointCloud, RigidTransform
from typing import Optional


def reshape_trajectory(
    traj,
    out_dim: int,
    fill_value: Optional[np.ndarray] = None,
    keep_indices: Optional[np.ndarray] = None,
    n_samples = 200
) -> PiecewisePolynomial:
    t_samples = np.linspace(traj.start_time(), traj.end_time(), n_samples)
    x_samples = np.array([traj.value(t).flatten() for t in t_samples]).T
    in_dim = x_samples.shape[0]

    if in_dim > out_dim:
        x_samples = x_samples[:out_dim, :]
        in_dim = out_dim

    # Determine keep indices
    if keep_indices is None:
        keep_indices = np.arange(in_dim)
    else:
        # Drop any indices >= in_dim or >= out_dim
        keep_indices = np.array([i for i in keep_indices if i < in_dim and i < out_dim])

    out = np.zeros((out_dim, len(t_samples)))

    # Insert the kept dimensions
    out[keep_indices, :] = x_samples[keep_indices, :]

    # Fill unused output dimensions
    if fill_value is None:
        fill_value = np.zeros(out_dim)
    fill_value = np.asarray(fill_value).flatten()
    unused = np.setdiff1d(np.arange(out_dim), keep_indices)
    out[unused, :] = fill_value[unused].reshape(-1, 1)

    return PiecewisePolynomial.CubicShapePreserving(t_samples, out)

def create_wsg_traj(iiwa_traj_end: float, wsg_start: float, wsg_trans: float,
                    wsg_end: float) -> CompositeTrajectory:
    eps = 1e-6
    t_end = iiwa_traj_end

    wsg_start = np.array([wsg_start]).reshape(1, 1)
    wsg_trans = np.array([wsg_trans]).reshape(1, 1)
    wsg_end = np.array([wsg_end]).reshape(1, 1)

    times = np.array([0.0, eps, t_end - eps, t_end])
    values = np.hstack([wsg_start, wsg_trans, wsg_trans, wsg_end])

    traj = PiecewisePolynomial.FirstOrderHold(times, values)
    return CompositeTrajectory([traj])

def concat_iiwa_traj(traj1: CompositeTrajectory, traj2: CompositeTrajectory) -> CompositeTrajectory:
    segments = []

    # Copy trajectory 1 segments unchanged
    for i in range(traj1.get_number_of_segments()):
        segments.append(traj1.segment(i))

    # Rebuild shifted segments of traj2
    dt = traj1.end_time() - traj2.start_time()
    times2 = traj2.get_segment_times()
    for i in range(traj2.get_number_of_segments()):
        seg = traj2.segment(i)
        ctrl = seg.control_points()
        t0 = times2[i]
        t1 = times2[i+1]

        shifted_seg = BezierCurve(t0 + dt, t1 + dt, ctrl)
        segments.append(shifted_seg)

    return CompositeTrajectory(segments)

def concat_wsg_traj(traj1, traj2):
    # Unwrap the underlying PiecewisePolynomial
    poly1 = traj1.segment(0)
    poly2 = traj2.segment(0)
    
    # Get the breakpoints and convert to numpy arrays
    t1 = np.array(poly1.get_segment_times())
    t2 = np.array(poly2.get_segment_times())
    
    # Time shift for traj2
    dt = t1[-1] - t2[0]
    new_t2 = t2 + dt
    
    # Collect all breakpoints and values
    all_times = np.concatenate([t1, new_t2[1:]])
    
    # Collect all values at breakpoints
    vals1 = np.hstack([poly1.value(t) for t in t1])
    vals2 = np.hstack([poly2.value(t) for t in t2])
    
    # Create new piecewise polynomial
    combined = PiecewisePolynomial.FirstOrderHold(all_times, np.hstack([vals1, vals2[:, 1:]]))
    
    return CompositeTrajectory([combined])

def get_pregrasp_pose(pose: RigidTransform) -> RigidTransform:
    return pose @ RigidTransform([0, -PREGRASP_OFFSET, 0])

def get_pc_from_depth(depth_img, intrinsics) -> PointCloud:
    fx = intrinsics.focal_x()
    fy = intrinsics.focal_y()
    cx = intrinsics.center_x()
    cy = intrinsics.center_y()

    H, W = depth_img.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    valid = depth_img > 0

    x = (u - cx) * depth_img / fx
    y = (v - cy) * depth_img / fy

    pc_cam = np.stack([x[valid], y[valid], depth_img[valid]], axis=1)

    pc = PointCloud(pc_cam.shape[0])
    pc.mutable_xyzs()[:] = pc_cam.T
    return pc

def transform_pointcloud(pc_in_cam: PointCloud, X_WC: RigidTransform) -> PointCloud:
    # Extract Nx3 array of XYZ points
    XYZ_cam = pc_in_cam.xyzs().T   # (N, 3)

    # Apply transform: p_W = R*p_C + t
    R = X_WC.rotation().matrix()
    t = X_WC.translation().reshape(1, 3)
    XYZ_world = (XYZ_cam @ R.T) + t

    # Create a new Drake PointCloud
    pc_world = PointCloud(XYZ_world.shape[0])
    pc_world.mutable_xyzs()[:] = XYZ_world.T
    return pc_world

def save_point_cloud(pc: PointCloud, path: str) -> None:
    xyz = pc.xyzs().T

    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(xyz)

    if pc.has_rgbs():
        rgb = pc.rgbs().T.astype(np.float32) / 255.0
        o3d_pc.colors = o3d.utility.Vector3dVector(rgb)

    o3d.io.write_point_cloud(path, o3d_pc, write_ascii=True)