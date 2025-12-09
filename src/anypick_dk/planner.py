from typing import List, Optional

import importlib.resources as resources
import logging
import numpy as np
from anypick_dk.constants import (
    IIWA_LEN, NUM_PICK_REGIONS, p_EETip, q_BotShelfPlace, q_BotShelfPre, q_Init, q_MidShelfPlace, q_MidShelfPre,
    q_TopShelfPlace, q_TopShelfPre, WSG_LEN, WSG_VEL_BOUND
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
    LoadIrisRegionsYamlFile,
    Point,
    RigidTransform,
    RotationMatrix,
    Solve,
)
Subgraph = GcsTrajectoryOptimization.Subgraph


class Planner:

    def __init__(self, sim_env: SimEnvironment):
        self.logger = logging.getLogger(__name__)

        self.sim_env = sim_env

        iris_regions_file = str(resources.files("anypick_dk") / "iris_regions" / "shelf_regions.yaml")
        self.iris_regions = LoadIrisRegionsYamlFile(iris_regions_file)

        gcs = GcsTrajectoryOptimization(IIWA_LEN + WSG_LEN)
        nodes = {
            "start_point": gcs.AddRegions(
                [Point(np.concat([q_Init, np.zeros(WSG_LEN)]))], order=0, name="start_point"
            ),
            "start": gcs.AddRegions(
                [self.iris_regions["start_region"]], order=1, name="start"
            ),
            "home": gcs.AddRegions(
                [self.iris_regions["home_region"]], order=1, name="home"
            ),
            "transition": gcs.AddRegions(
                [self.iris_regions["transition_region"]], order=1, name="transition"
            ),
            "top_shelf": gcs.AddRegions(
                [self.iris_regions["top_shelf_region"]], order=1, name="top_shelf"
            ),
            "top_shelf_approach": gcs.AddRegions(
                [self.iris_regions["top_shelf_approach_region"]], order=1, name="top_shelf_approach"
            ),
            "mid_shelf": gcs.AddRegions(
                [self.iris_regions["mid_shelf_region"]], order=1, name="mid_shelf"
            ),
            "mid_shelf_approach": gcs.AddRegions(
                [self.iris_regions["mid_shelf_approach_region"]], order=1, name="mid_shelf_approach"
            ),
            "bot_shelf": gcs.AddRegions(
                [self.iris_regions["bot_shelf_region"]], order=1, name="bot_shelf_region"
            ),
            "bot_shelf_approach": gcs.AddRegions(
                [self.iris_regions["bot_shelf_approach_region"]], order=1, name="bot_shelf_approach"
            ),
            "object": gcs.AddRegions(
                [self.iris_regions["object_region"]], order=1, name="object"
            ),
            "top_shelf_place": gcs.AddRegions(
                [Point(np.concat([q_TopShelfPlace, np.zeros(WSG_LEN)]))], order=0, name="top_shelf_place"
            ),
            "mid_shelf_place": gcs.AddRegions(
                [Point(np.concat([q_MidShelfPlace, np.zeros(WSG_LEN)]))], order=0, name="mid_shelf_place"
            ),
            "bot_shelf_place": gcs.AddRegions(
                [Point(np.concat([q_BotShelfPlace, np.zeros(WSG_LEN)]))], order=0, name="bot_shelf_place"
            ),
            "top_shelf_pre": gcs.AddRegions(
                [Point(np.concat([q_TopShelfPre, np.zeros(WSG_LEN)]))], order=0, name="top_shelf_pre"
            ),
            "mid_shelf_pre": gcs.AddRegions(
                [Point(np.concat([q_MidShelfPre, np.zeros(WSG_LEN)]))], order=0, name="mid_shelf_pre"
            ),
            "bot_shelf_pre": gcs.AddRegions(
                [Point(np.concat([q_BotShelfPre, np.zeros(WSG_LEN)]))], order=0, name="bot_shelf_pre"
            ),
        }

        for i in range(NUM_PICK_REGIONS):
            nodes[f"pick{i}"] = gcs.AddRegions(
                [self.iris_regions[f"pick_region_{i}"]], order=1, name=f"pick{i}"
            )

        gcs.AddEdges(nodes["start_point"], nodes["start"])
        gcs.AddEdges(nodes["start"], nodes["home"])

        gcs.AddEdges(nodes["home"], nodes["object"])
        gcs.AddEdges(nodes["object"], nodes["home"])

        for i in range(NUM_PICK_REGIONS):
            gcs.AddEdges(nodes["object"], nodes[f"pick{i}"])
            gcs.AddEdges(nodes[f"pick{i}"], nodes["object"])

        gcs.AddEdges(nodes["home"], nodes["top_shelf_approach"])
        gcs.AddEdges(nodes["top_shelf_approach"], nodes["home"])

        gcs.AddEdges(nodes["home"], nodes["mid_shelf_approach"])
        gcs.AddEdges(nodes["mid_shelf_approach"], nodes["home"])

        gcs.AddEdges(nodes["home"], nodes["transition"])
        gcs.AddEdges(nodes["transition"], nodes["home"])

        gcs.AddEdges(nodes["transition"], nodes["bot_shelf_approach"])
        gcs.AddEdges(nodes["bot_shelf_approach"], nodes["transition"])

        gcs.AddEdges(nodes["top_shelf_approach"], nodes["top_shelf"])
        gcs.AddEdges(nodes["top_shelf"], nodes["top_shelf_approach"])

        gcs.AddEdges(nodes["mid_shelf_approach"], nodes["mid_shelf"])
        gcs.AddEdges(nodes["mid_shelf"], nodes["mid_shelf_approach"])

        gcs.AddEdges(nodes["bot_shelf_approach"], nodes["bot_shelf"])
        gcs.AddEdges(nodes["bot_shelf"], nodes["bot_shelf_approach"])

        gcs.AddEdges(nodes["top_shelf_approach"], nodes["top_shelf_pre"])
        gcs.AddEdges(nodes["top_shelf_pre"], nodes["top_shelf_approach"])

        gcs.AddEdges(nodes["mid_shelf_approach"], nodes["mid_shelf_pre"])
        gcs.AddEdges(nodes["mid_shelf_pre"], nodes["mid_shelf_approach"])

        gcs.AddEdges(nodes["bot_shelf_approach"], nodes["bot_shelf_pre"])
        gcs.AddEdges(nodes["bot_shelf_pre"], nodes["bot_shelf_approach"])

        gcs.AddEdges(nodes["top_shelf"], nodes["top_shelf_place"])
        gcs.AddEdges(nodes["top_shelf_place"], nodes["top_shelf"])

        gcs.AddEdges(nodes["mid_shelf"], nodes["mid_shelf_place"])
        gcs.AddEdges(nodes["mid_shelf_place"], nodes["mid_shelf"])

        gcs.AddEdges(nodes["bot_shelf"], nodes["bot_shelf_place"])
        gcs.AddEdges(nodes["bot_shelf_place"], nodes["bot_shelf"])

        gcs.AddTimeCost()
        lb = self.sim_env.plant.GetVelocityLowerLimits()[:IIWA_LEN + WSG_LEN] / 8
        lb[~np.isfinite(lb)] = -WSG_VEL_BOUND
        ub = self.sim_env.plant.GetVelocityUpperLimits()[:IIWA_LEN + WSG_LEN] / 8
        ub[~np.isfinite(ub)] = WSG_VEL_BOUND
        gcs.AddVelocityBounds(lb, ub)

        self.gcs = gcs
        self.nodes = nodes

    def solve_fk(self) -> RigidTransform:
        X_WE = self.sim_env.plant.CalcRelativeTransform(
            self.sim_env.plant_context,
            self.sim_env.plant.world_frame(),
            self.sim_env.ee_frame,
        )
        X_ETip = RigidTransform(p_EETip)
        return X_WE @ X_ETip

    def solve_ik(self, tf: RigidTransform, q0: np.ndarray = np.zeros(IIWA_LEN),
                 trans_tol: float = 0.0, ang_tol: float = 0.0) -> Optional[np.ndarray]:
        diagram_context = self.sim_env.diagram_context.Clone()
        plant_context = self.sim_env.plant.GetMyContextFromRoot(diagram_context)

        ik = InverseKinematics(self.sim_env.plant, plant_context)

        tf_min = tf.translation() - np.ones(3) * trans_tol
        tf_max = tf.translation() + np.ones(3) * trans_tol
        ik.AddPositionConstraint(self.sim_env.ee_frame, p_EETip, self.sim_env.plant.world_frame(),
                                 tf_min, tf_max)

        ik.AddOrientationConstraint(self.sim_env.plant.world_frame(), tf.rotation(),
                                    self.sim_env.ee_frame, RotationMatrix(), ang_tol)

        q_vars = ik.q()[:IIWA_LEN]
        prog = ik.prog()
        prog.AddQuadraticErrorCost(1, q0, q_vars)
        prog.SetInitialGuess(q_vars, q0)

        result = Solve(prog)
        if not result.is_success():
            self.logger.error("IK failed!")
            return None

        self.logger.info("IK success.")
        return result.GetSolution(q_vars)

    def solve_ik_pos(self, pos: List, q0: np.ndarray = np.zeros(IIWA_LEN),
                     ik_min_dist: float = 0.01, ik_influence_dist: float = 0.05) -> Optional[np.ndarray]:
        diagram_context = self.sim_env.diagram_context.Clone()
        plant_context = self.sim_env.plant.GetMyContextFromRoot(diagram_context)

        ik = InverseKinematics(self.sim_env.plant, plant_context)
        ik.AddMinimumDistanceLowerBoundConstraint(ik_min_dist, ik_influence_dist)
        ik.AddPositionConstraint(self.sim_env.ee_frame, p_EETip, self.sim_env.plant.world_frame(), pos, pos)

        q_vars = ik.q()[:IIWA_LEN]
        prog = ik.prog()
        prog.AddQuadraticErrorCost(1, q0, q_vars)
        prog.SetInitialGuess(q_vars, q0)

        result = Solve(ik.prog())
        if not result.is_success():
            self.logger.error("IK failed!")
            return None

        self.logger.info("IK success.")
        return result.GetSolution(q_vars)

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
            self.logger.error("No feasible path found!")
            return None

        self.logger.info("GCS path successfully found.")
        return traj
