import logging
import operator
import py_trees

from anypick_dk.behaviors import (
    AskForPrompt, ConfirmDetection, ExecuteTrajectory, GroundedSamDetect, PlanTrajectory, EndBehavior
)


class Orchestrator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.tree = self._create_tree()
        self.tree.setup(timeout=15)
    
    def _create_tree(self):
        bb = py_trees.blackboard.Client(name="initializer")
        bb.register_key("current_prompt", access=py_trees.common.Access.WRITE)
        bb.register_key("detected_object", access=py_trees.common.Access.WRITE)
        bb.register_key("completed_detections", access=py_trees.common.Access.WRITE)

        bb.current_prompt = ""
        bb.detected_object = ""
        bb.completed_detections = 0

        check_done = py_trees.behaviours.CheckBlackboardVariableValue(
            name="is_objects_detected",
            check=py_trees.common.ComparisonExpression(
                variable="completed_detections",
                value=3,
                operator=operator.ge
            )
        )

        detect_sequence = py_trees.composites.Sequence(name="detect_one_object", memory=True)
        detect_sequence.add_children([
            AskForPrompt(),
            GroundedSamDetect(),
            ConfirmDetection(),
        ])

        retry_on_detect_fail = py_trees.decorators.Retry(
            name="retry_on_detect_fail",
            child=detect_sequence,
            num_failures=3
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