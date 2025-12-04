import logging
import py_trees

logger = logging.getLogger(__name__)


class AskForPrompt(py_trees.behaviour.Behaviour):
    def __init__(self, name="ask_for_prompt"):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key("current_prompt", access=py_trees.common.Access.WRITE)

    def update(self):
        prompt = input("\nEnter GroundingDINO prompt to detect: ")
        self.blackboard.current_prompt = prompt
        return py_trees.common.Status.SUCCESS


class ConfirmDetection(py_trees.behaviour.Behaviour):
    def __init__(self, name="confirm_detection"):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key("completed_detections", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("detected_object", access=py_trees.common.Access.READ)

    def update(self):
        detected = self.blackboard.detected_object

        print(f"Detected object = '{detected}'")
        user = input("Is this correct? (y/n): ").strip().lower()

        if user == 'y':
            self.blackboard.completed_detections += 1
            print(f"Detection confirmed. Total = {self.blackboard.completed_detections}")
            return py_trees.common.Status.SUCCESS
        else:
            print("Detection failed.")
            return py_trees.common.Status.FAILURE


class ExecuteTrajectory(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("execute_trajectory")

    def update(self):
        print("Executing trajectory...")
        print("Trajectory executed successfully!")
        return py_trees.common.Status.SUCCESS


class GroundedSamDetect(py_trees.behaviour.Behaviour):
    def __init__(self, name="grounded_sam_detect"):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key("current_prompt", access=py_trees.common.Access.READ)
        self.blackboard.register_key("detected_object", access=py_trees.common.Access.WRITE)

    def update(self):
        prompt = self.blackboard.current_prompt

        if not prompt:
            detected = "nothing"
        else:
            detected = prompt.strip().split()[-1].lower()

        print(f"Mock detection result: '{detected}'")
        self.blackboard.detected_object = detected
        return py_trees.common.Status.SUCCESS


class PlanTrajectory(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("plan_trajectory")

    def update(self):
        print("\nPlanning full trajectory based on detected objects...")
        return py_trees.common.Status.SUCCESS


class EndBehavior(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("end_behavior")
    
    def update(self):
        input("\nPress any key to end tree")
        return py_trees.common.Status.SUCCESS