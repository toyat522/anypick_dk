import logging
import operator
import py_trees

from anypick_dk.behaviors import (
    ExecuteTrajectory, GetGraspPose, GroundedSamDetect,
    PlanTrajectory, EndBehavior
)
from anypick_dk.constants import MAX_FAILURES, NUM_DETECTIONS
from anypick_dk.grounded_sam_wrapper import GroundedSamWrapper
from anypick_dk.planner import Planner
from anypick_dk.sim_environment import SimEnvironment


class Orchestrator:
    def __init__(self, scenario_file: str):
        self.logger = logging.getLogger(__name__)

        self.sim_env = SimEnvironment(scenario_file)
        self.planner = Planner(self.sim_env)
        self.gdsam = GroundedSamWrapper()

        self.tree = self._create_tree()
        self.tree.setup()
    
    def _create_tree(self):
        bb = py_trees.blackboard.Client(name="initializer")
        bb.register_key("num_detections", access=py_trees.common.Access.WRITE)
        bb.register_key("depth_images", access=py_trees.common.Access.WRITE)
        bb.register_key("detected_object", access=py_trees.common.Access.WRITE)
        bb.register_key("masks", access=py_trees.common.Access.WRITE)
        bb.num_detections = 0
        bb.depth_images = []
        bb.detected_object = ""
        bb.masks = []

        check_done = py_trees.behaviours.CheckBlackboardVariableValue(
            name="is_objects_detected",
            check=py_trees.common.ComparisonExpression(
                variable="num_detections",
                value=NUM_DETECTIONS,
                operator=operator.ge
            )
        )

        detect_sequence = py_trees.composites.Sequence(name="detect_one_object", memory=True)
        detect_sequence.add_children([
            GroundedSamDetect(self.sim_env, self.gdsam),
            GetGraspPose(self.sim_env)
        ])

        retry_on_detect_fail = py_trees.decorators.Retry(
            name="retry_on_detect_fail",
            child=detect_sequence,
            num_failures=MAX_FAILURES
        )

        loop_detect = py_trees.decorators.SuccessIsRunning(
            name="loop_detect",
            child=retry_on_detect_fail
        )

        gather_phase = py_trees.composites.Selector(name="detect_all_objects", memory=False)
        gather_phase.add_children([check_done, loop_detect])

        finish_sequence = py_trees.composites.Sequence(name="plan_and_execute_trajectory", memory=False)
        finish_sequence.add_children([
            PlanTrajectory(),
            ExecuteTrajectory(),
            EndBehavior()
        ])

        root = py_trees.composites.Sequence(name="root", memory=True)
        root.add_children([gather_phase, finish_sequence])

        return py_trees.trees.BehaviourTree(root)

    def tick(self) -> py_trees.common.Status:
        self.tree.tick()

        self.logger.info(py_trees.display.unicode_tree(
            root=self.tree.root, 
            show_status=True
        ))
        
        return self.tree.root.status