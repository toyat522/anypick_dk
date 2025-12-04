import cv2
import numpy as np
import py_trees

from anypick_dk.grounded_sam_wrapper import GroundedSamWrapper
from anypick_dk.constants import NUM_CAMERAS, VOXEL_SIZE
from anypick_dk.planner import Planner
from anypick_dk.sim_environment import SimEnvironment
from anypick_dk.utils import get_pc_from_depth, transform_pointcloud
from pydrake.all import Concatenate, Rgba


class EndBehavior(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("end_behavior")
    
    def update(self):
        input("\nPress the enter key to end tree: ")
        return py_trees.common.Status.SUCCESS


class ExecuteTrajectory(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("execute_trajectory")

    def update(self):
        self.logger.info("Executing trajectory...")



        self.logger.info("Trajectory executed successfully!")
        return py_trees.common.Status.SUCCESS


class GetGraspPose(py_trees.behaviour.Behaviour):
    def __init__(self, sim_env: SimEnvironment):
        name = "get_grasp_pose"
        super().__init__(name)
        self.sim_env = sim_env
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key("masks", access=py_trees.common.Access.READ)
    
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
        
        # TODO: actually run grasp pose generation
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


class PlanTrajectory(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("plan_trajectory")

    def update(self):
        self.logger.info("\nPlanning full trajectory based on detected objects...")
        return py_trees.common.Status.SUCCESS