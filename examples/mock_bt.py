import operator
import py_trees


class AskForPrompt(py_trees.behaviour.Behaviour):
    def __init__(self):
        name = "ask_for_prompt"
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key("current_prompt", access=py_trees.common.Access.WRITE)

    def update(self):
        prompt = input(f"\nEnter GroundingDINO prompt to detect: ")
        self.blackboard.current_prompt = prompt
        return py_trees.common.Status.SUCCESS


class MockGroundedSAM(py_trees.behaviour.Behaviour):
    def __init__(self):
        name = "grounded_sam"
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


class AskUserConfirmation(py_trees.behaviour.Behaviour):
    def __init__(self):
        name = "confirm_detection"
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


class PlanTrajectory(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("plan_trajectory")

    def update(self):
        print("\nPlanning full trajectory based on detected objects...")
        return py_trees.common.Status.SUCCESS


class ExecuteTrajectory(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("execute_trajectory")

    def update(self):
        print("Executing trajectory...")
        print("Trajectory executed successfully!")
        return py_trees.common.Status.SUCCESS


class EndBehavior(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("end_behavior")
    
    def update(self):
        input("\nPress the enter key to end tree: ")
        return py_trees.common.Status.SUCCESS


def create_tree():
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
        MockGroundedSAM(),
        AskUserConfirmation(),
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


def main():
    tree = create_tree()
    tree.setup()

    py_trees.display.render_dot_tree(tree.root)

    print("\n=== Behavior Tree Started ===")
    print("Goal: Detect 3 objects, then plan.")

    while True:
        tree.tick()

        print(py_trees.display.unicode_tree(
            root=tree.root, 
            show_status=True
        ))
        
        if tree.root.status == py_trees.common.Status.SUCCESS:
            print("\nTree finished successfully.")
            break
        if tree.root.status == py_trees.common.Status.FAILURE:
            print("\nTree failed.")
            break


if __name__ == "__main__":
    main()
