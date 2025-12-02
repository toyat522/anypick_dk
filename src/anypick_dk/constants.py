import numpy as np

IIWA_LEN = 7
IIWA_MAX_TORQUE = 8
p_EETip = np.array([0, 0.1, 0])
q_TopShelfPlace = np.array([0, 0.49903189, 0, -0.99302722, 0, 0.35085392, 0])
q_MidShelfPlace = np.array([0, 0.67953517, 0, -1.49155983, 0, -0.50557705, 0])
q_BotShelfPlace = np.array([0, 1.15303645, 0, -1.42083483, 0, -0.94480558, 0])
q_Object = [1.02, 0.98, 0.32, -1.09, -0.12, 1.1, 0.9]
SIM_END_SECS = 1.0
WSG_CLOSED = 0.0
WSG_LEN = 2
WSG_OPENED = 0.1
WSG_VEL_BOUND = 0.5