from typing import List, Optional

import logging
import numpy as np
from anypick_dk.constants import (
    IIWA_LEN,
    p_EETip
)
from anypick_dk.sim_environment import SimEnvironment
from pydrake.all import (
    CompositeTrajectory,
    GcsTrajectoryOptimization,
    GraphOfConvexSetsOptions,
    HPolyhedron,
    InverseKinematics,
    IrisInConfigurationSpace,
    IrisOptions,
    Solve,
)
Subgraph = GcsTrajectoryOptimization.Subgraph


class Planner:

    gcs: Optional[GcsTrajectoryOptimization] = None

    def __init__(self, sim_env: SimEnvironment):
        self.sim_env = sim_env

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def solve_ik_pos(self, pos: List, q0: np.ndarray = np.zeros(IIWA_LEN),
                     ik_min_dist: float = 0.01, ik_influence_dist: float = 0.05) -> Optional[np.ndarray]:
        ik = InverseKinematics(self.sim_env.plant, self.sim_env.plant_context)
        ik.AddMinimumDistanceLowerBoundConstraint(ik_min_dist, ik_influence_dist)
        ik.AddPositionConstraint(self.sim_env.ee_frame, p_EETip, self.sim_env.plant.world_frame(), pos, pos)

        q_vars = ik.q()[:IIWA_LEN]
        prog = ik.prog()
        prog.AddQuadraticErrorCost(1, q0, q_vars)
        prog.SetInitialGuess(q_vars, q0)

        result = Solve(ik.prog())
        if not result.is_success():
            self.logger.error("IK failed")
            return None

        self.logger.info("IK success")
        return result.GetSolution(q_vars)

    def set_base_gcs(self, gcs: GcsTrajectoryOptimization) -> None:
        self.gcs = gcs

    def create_iris_region(self, q_seed: np.ndarray, iris_cspace_margin: float = 0.02) -> HPolyhedron:
        # Save initial plant position
        q0 = self.sim_env.get_iiwa_position()

        self.sim_env.set_iiwa_position(q_seed)
        self.sim_env.diagram.ForcedPublish(self.sim_env.diagram_context)
        options = IrisOptions()
        options.num_collision_infeasible_samples = 3
        options.require_sample_point_is_contained = True
        options.configuration_space_margin = iris_cspace_margin 
        region = IrisInConfigurationSpace(self.sim_env.plant, self.sim_env.plant_context, options)

        # Restore initial plant position
        self.sim_env.set_iiwa_position(q0)
        return region

    def solve_gcs(self, source: Subgraph, target: Subgraph) -> Optional[CompositeTrajectory]:
        assert self.gcs is not None, "Planner needs to call set_base_gcs first"

        options = GraphOfConvexSetsOptions()
        options.preprocessing = True
        options.max_rounded_paths = 10

        traj, result = self.gcs.SolvePath(source, target, options)

        if not result.is_success():
            self.logger.error("No feasible path found")
            return None

        self.logger.info("GCS path successfully found")
        return traj
