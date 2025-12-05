import numpy as np

from pydrake.all import RigidTransform, RollPitchYaw

IIWA_LEN = 7
IIWA_MAX_TORQUE = 8
MAX_FAILURES = 3
NUM_CAMERAS = 2
NUM_DETECTIONS = 3
PREGRASP_Z = 0.1
p_EETip = np.array([0, 0.1, 0])
q_BotShelfPlace = np.array([0, 1.15303645, 0, -1.42083483, 0, -0.94480558, 0])
q_BotShelfPre = np.array([0, 0.82594577, 0, -1.96760834, 0, -1.16446792, 0])
q_Init = np.array([0, 0.1, 0, -1.2, 0, 1.6, 0])
q_MidShelfPlace = np.array([0, 0.67953517, 0, -1.49155983, 0, -0.50557705, 0])
q_MidShelfPre = np.array([0, 0.26447518, 0, -1.92140087, 0, -0.52034149, 0])
q_Object = np.array([-2.07096646, -1.30205044, -0.17537454, 0.59861396, 0.17821979, -1.24971752, 0.18142082])
q_TopShelfPlace = np.array([0, 0.56044816, 0, -0.90046223, 0, 0.38197154, 0])
q_TopShelfPre = np.array([0, 0.10978964, 0, -1.32262024, 0, 0.41050794, 0])
SIM_END_SECS = 1.0

X_WCan = RigidTransform(
    RollPitchYaw(np.deg2rad(np.array([-90, 0, -45]))),
    [0.40, 0.65, 0.05]
)
X_WGelatin = RigidTransform(
    RollPitchYaw(np.deg2rad(np.array([-90, 0, 50]))),
    [0.05, 0.75, 0.045]
)
X_WMustard = RigidTransform(
    RollPitchYaw(np.deg2rad(np.array([-90, 0, 75]))),
    [-0.08, 0.56, 0.14]
)
X_WSugar = RigidTransform(
    RollPitchYaw(np.deg2rad(np.array([-87, 0, 115]))),
    [0.22, 0.65, 0.115]
)

VOXEL_SIZE = 0.005
WSG_CLOSED = 0.0
WSG_LEN = 2
WSG_OPENED = 0.1
WSG_VEL_BOUND = 0.5