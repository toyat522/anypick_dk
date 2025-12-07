import cv2
import numpy as np
import py_trees

from anypick_dk.grounded_sam_wrapper import GroundedSamWrapper
from anypick_dk.constants import (
    NUM_CAMERAS, q_Object, VOXEL_SIZE, WSG_CLOSED, WSG_LEN, WSG_OPENED
)
from anypick_dk.planner import Planner
from anypick_dk.sim_environment import SimEnvironment
from anypick_dk.utils import (
    concat_iiwa_traj, concat_wsg_traj, create_wsg_traj, get_pc_from_depth, transform_pointcloud,
    save_point_cloud
)
from functools import reduce
from pydrake.all import Concatenate, Point, Rgba, RigidTransform


class EndBehavior(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("end_behavior")

    def update(self):
        input("\nPress the enter key to end tree: ")
        return py_trees.common.Status.SUCCESS


class GetGraspPose(py_trees.behaviour.Behaviour):
    def __init__(self, sim_env: SimEnvironment):
        name = "get_grasp_pose"
        super().__init__(name)
        self.sim_env = sim_env
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key("masks", access=py_trees.common.Access.READ)
        self.blackboard.register_key("poses", access=py_trees.common.Access.WRITE)

    def update(self):
        # Get point clouds in world frame from masks and depth images
        point_clouds = []
        for i, mask in enumerate(self.blackboard.masks):
            intrinsics = self.sim_env.get_camera_intrinsics(i)
            depth_img = self.sim_env.get_camera_depth(i)
            masked_depth = depth_img.copy()
            masked_depth[~mask] = 0.0
            point_clouds.append(
                transform_pointcloud(
                    get_pc_from_depth(masked_depth, intrinsics),
                    self.sim_env.get_camera_pose(i)
                )
            )
        obj_pc = Concatenate(point_clouds)
        obj_pc = obj_pc.VoxelizedDownSample(voxel_size=VOXEL_SIZE)
        self.sim_env.meshcat.SetObject(
            "obj_pc", obj_pc, point_size=0.01, rgba=Rgba(1, 0, 0)
        )

        print("\nThe object point cloud is visualized in meshcat.")
        user = input("Is the object point cloud valid? (y/n): ").strip().lower()

        self.sim_env.meshcat.Delete("obj_pc")
        if user != 'y':
            self.logger.warning("Point cloud extraction failed!")
            return py_trees.common.Status.FAILURE
        self.logger.info("Point cloud extraction succeeded.")
        save_point_cloud(obj_pc, "obj_pc.ply")

        # TODO: actually run grasp pose generation
        pose = RigidTransform()
        self.blackboard.poses.append(pose)
        self.sim_env.visualize_frame("obj_frame", pose)

        print("\nThe grasp pose is visualized in meshcat.")
        user = input("Is the grasp pose valid? (y/n): ").strip().lower()

        if user != 'y':
            self.sim_env.clear_frame("obj_frame")
            self.logger.warning("Grasp pose calculation failed!")
            return py_trees.common.Status.FAILURE
        self.logger.info("Grasp pose calculation succeeded.")

        return py_trees.common.Status.SUCCESS


class GetGTGraspPose(py_trees.behaviour.Behaviour):
    def __init__(self, sim_env: SimEnvironment, gt_poses: RigidTransform):
        name = "get_grasp_pose"
        super().__init__(name)
        self.sim_env = sim_env
        self.gt_poses = gt_poses
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key("num_detections", access=py_trees.common.Access.READ)
        self.blackboard.register_key("poses", access=py_trees.common.Access.WRITE)
    
    def update(self):
        pose = self.gt_poses[self.blackboard.num_detections - 1]
        self.sim_env.visualize_frame("obj_frame", pose)

        print("\nThe grasp pose is visualized in meshcat.")
        user = input("Is the grasp pose valid? (y/n): ").strip().lower()

        if user != 'y':
            self.sim_env.clear_frame("obj_frame")
            self.logger.warning("Grasp pose calculation failed!")
            return py_trees.common.Status.FAILURE
        self.logger.info("Grasp pose calculation succeeded.")

        self.blackboard.poses.append(pose)
        return py_trees.common.Status.SUCCESS


class GroundedSamDetect(py_trees.behaviour.Behaviour):
    def __init__(self, sim_env: SimEnvironment, gdsam: GroundedSamWrapper):
        name = "grounded_sam_detect"
        super().__init__(name)
        self.sim_env = sim_env
        self.gdsam = gdsam
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key("masks", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("num_detections", access=py_trees.common.Access.WRITE)

    def update(self):
        prompt = input("\nEnter GroundingDINO prompt to detect: ")

        masks = []
        for i in range(NUM_CAMERAS):
            img = self.sim_env.get_camera_bgr(i)

            self.logger.info(f"Running GroundedSAM on camera{i} with prompt '{prompt}'.")
            _, mask = self.gdsam.detect_and_segment(img, [prompt])

            if mask is None:
                self.logger.warning(f"Failed to detect '{prompt}' on camera{i}!")
                return py_trees.common.Status.FAILURE

            print("\nPress 'q' to continue.")
            cv2.imshow("annotated image", self.gdsam.annotate())
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()

            user = input("Is the detection valid? (y/n): ").strip().lower()
            if user != 'y':
                self.logger.warning("Detection failed!")
                return py_trees.common.Status.FAILURE
            
            mask = cv2.erode(
                mask.astype(np.uint8) * 255, np.ones((7, 7), np.uint8), iterations=1
            ).astype(bool)
            masks.append(mask)

        self.logger.info(f"Detection confirmed. Total number of detections: {self.blackboard.num_detections}.")
        self.blackboard.num_detections += 1
        self.blackboard.masks = masks
        return py_trees.common.Status.SUCCESS


class PlanAndExecuteTrajectory(py_trees.behaviour.Behaviour):
    def __init__(self, sim_env: SimEnvironment, planner: Planner):
        name = "plan_trajectory"
        super().__init__(name)
        self.sim_env = sim_env
        self.planner = planner
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key("poses", access=py_trees.common.Access.READ)

    def update(self):
        self.logger.info("\nPlanning trajectory based on detected objects...")

        prev_end = self.planner.nodes["start_point"]
        pres = [
            self.planner.nodes["top_shelf_pre"],
            self.planner.nodes["mid_shelf_pre"],
            self.planner.nodes["bot_shelf_pre"],
        ]
        places = [
            self.planner.nodes["top_shelf_place"],
            self.planner.nodes["mid_shelf_place"],
            self.planner.nodes["bot_shelf_place"],
        ]

        iiwa_trajs = []
        wsg_trajs = []
        for i, pose in enumerate(self.blackboard.poses):

            q_obj = self.planner.solve_ik(pose, q_Object)
            if q_obj is None:
                self.logger.warning("Ending plan due to IK failure.")
                return py_trees.common.Status.FAILURE

            obj_node = self.planner.gcs.AddRegions(
                [Point(np.concat([q_obj, np.zeros(WSG_LEN)]))], order=0, name=f"obj{i}"
            )
            self.planner.gcs.AddEdges(obj_node, self.planner.nodes["pick0"])
            self.planner.gcs.AddEdges(self.planner.nodes["pick0"], obj_node)
            self.planner.gcs.AddEdges(obj_node, self.planner.nodes["pick1"])
            self.planner.gcs.AddEdges(self.planner.nodes["pick1"], obj_node)

            iiwa_trajs.append(self.planner.solve_gcs(prev_end, obj_node))
            if iiwa_trajs[-1] is None:
                self.logger.warning("Ending plan due to GCS failure from path to obj.")
                return py_trees.common.Status.FAILURE

            wsg_trajs.append(create_wsg_traj(iiwa_trajs[-1].end_time(), WSG_OPENED, WSG_OPENED, WSG_CLOSED))

            iiwa_trajs.append(self.planner.solve_gcs(obj_node, pres[i]))
            if iiwa_trajs[-1] is None:
                self.logger.warning("Ending plan due to GCS failure from obj to pre-place.")
                return py_trees.common.Status.FAILURE

            wsg_trajs.append(create_wsg_traj(iiwa_trajs[-1].end_time(), WSG_CLOSED, WSG_CLOSED, WSG_CLOSED))

            iiwa_trajs.append(self.planner.solve_gcs(pres[i], places[i]))
            if iiwa_trajs[-1] is None:
                self.logger.warning("Ending plan due to GCS failure from pre-place to place.")
                return py_trees.common.Status.FAILURE

            wsg_trajs.append(create_wsg_traj(iiwa_trajs[-1].end_time(), WSG_CLOSED, WSG_CLOSED, WSG_OPENED))
            prev_end = places[i]
        
        full_iiwa_traj = reduce(concat_iiwa_traj, iiwa_trajs)
        full_wsg_traj = reduce(concat_wsg_traj, wsg_trajs)

        self.logger.info("Trajectory planning success! Simulating...")

        self.sim_env.build_diagram_with_controller(full_iiwa_traj, full_wsg_traj)
        self.sim_env.simulate()
        self.logger.info("Simulation complete!")

        return py_trees.common.Status.SUCCESS