import py_trees
import operator
import sys

# ============================================================
# Node 1: Ask User for a GroundingDINO prompt
# ============================================================
class AskForPrompt(py_trees.behaviour.Behaviour):
    def __init__(self, name="AskForPrompt"):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key("current_prompt", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("current_object_index", access=py_trees.common.Access.READ)
        self.blackboard.register_key("object_list", access=py_trees.common.Access.READ)

    def update(self):
        # Index safety check
        i = self.blackboard.current_object_index
        if i >= len(self.blackboard.object_list):
            return py_trees.common.Status.FAILURE
            
        target_obj = self.blackboard.object_list[i]

        try:
            # We use a simple input here (blocking) for the mock
            prompt = input(f"\n👉 Enter GroundingDINO prompt to detect '{target_obj}': ")
            self.blackboard.current_prompt = prompt
            return py_trees.common.Status.SUCCESS
        except EOFError:
            return py_trees.common.Status.FAILURE

# ============================================================
# Node 2: Mock Object Detection
# ============================================================
class MockGroundingDINO(py_trees.behaviour.Behaviour):
    def __init__(self, name="GroundingDINO"):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key("current_prompt", access=py_trees.common.Access.READ)
        self.blackboard.register_key("detected_object", access=py_trees.common.Access.WRITE)

    def update(self):
        prompt = self.blackboard.current_prompt
        # Mock logic: "detect" the last word of the prompt
        if not prompt:
            detected = "nothing"
        else:
            detected = prompt.strip().split()[-1].lower()

        print(f"🔍 Mock detection result: '{detected}'")
        self.blackboard.detected_object = detected
        return py_trees.common.Status.SUCCESS

# ============================================================
# Node 3: Ask user if detection is correct
# ============================================================
class AskUserConfirmation(py_trees.behaviour.Behaviour):
    def __init__(self, name="ConfirmDetection"):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key("detected_object", access=py_trees.common.Access.READ)
        self.blackboard.register_key("object_list", access=py_trees.common.Access.READ)
        self.blackboard.register_key("current_object_index", access=py_trees.common.Access.READ)
        self.blackboard.register_key("detection_confirmed", access=py_trees.common.Access.WRITE)

    def update(self):
        true_object = self.blackboard.object_list[self.blackboard.current_object_index]
        detected = self.blackboard.detected_object

        print(f"📸 Detected object = '{detected}' (expected '{true_object}')")
        user = input("Is this correct? (y/n): ").strip().lower()
        
        self.blackboard.detection_confirmed = (user == "y")
        return py_trees.common.Status.SUCCESS

# ============================================================
# Node 4: Validate & Increment
# ============================================================
class ValidateDetection(py_trees.behaviour.Behaviour):
    def __init__(self, name="ValidateDetection"):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key("detection_confirmed", access=py_trees.common.Access.READ)
        self.blackboard.register_key("current_object_index", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key("completed_detections", access=py_trees.common.Access.WRITE)

    def update(self):
        if self.blackboard.detection_confirmed:
            self.blackboard.completed_detections += 1
            self.blackboard.current_object_index += 1
            print(f"✅ Detection confirmed. Total = {self.blackboard.completed_detections}")
            return py_trees.common.Status.SUCCESS
        else:
            print("❌ Detection incorrect — retrying same object.")
            # We return FAILURE here. The sequence will fail, 
            # but our Root Selector will simply retry the loop.
            return py_trees.common.Status.FAILURE

# ============================================================
# Planning & Execution Nodes
# ============================================================
class PlanTrajectory(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("PlanTrajectory")
    def update(self):
        print("\n📐 Planning full trajectory based on 3 objects...")
        return py_trees.common.Status.SUCCESS

class ExecuteTrajectory(py_trees.behaviour.Behaviour):
    def __init__(self):
        super().__init__("ExecuteTrajectory")
    def update(self):
        print("🤖 Executing trajectory...")
        print("🎉 Mission Complete!")
        return py_trees.common.Status.SUCCESS

# ============================================================
# Constructing the Behavior Tree
# ============================================================
def create_tree():
    # 1. Initialize Blackboard
    bb = py_trees.blackboard.Client(name="Init")
    bb.register_key("object_list", access=py_trees.common.Access.WRITE)
    bb.register_key("current_object_index", access=py_trees.common.Access.WRITE)
    bb.register_key("current_prompt", access=py_trees.common.Access.WRITE)
    bb.register_key("detected_object", access=py_trees.common.Access.WRITE)
    bb.register_key("detection_confirmed", access=py_trees.common.Access.WRITE)
    bb.register_key("completed_detections", access=py_trees.common.Access.WRITE)

    bb.object_list = ["apple", "mug", "bottle", "banana"]
    bb.current_object_index = 0
    bb.current_prompt = ""
    bb.detected_object = ""
    bb.detection_confirmed = False
    bb.completed_detections = 0

    # 2. Check Done Condition (The Fix is Here)
    # We use ComparisonExpression for py_trees 2.0+
    check_done = py_trees.behaviours.CheckBlackboardVariableValue(
        name="Have We Found 3?",
        check=py_trees.common.ComparisonExpression(
            variable="completed_detections",
            value=3,
            operator=operator.ge
        )
    )

    # 3. Define the Detection Sequence (One single pass)
    # If ValidateDetection fails (user says no), this entire sequence fails.
    detect_sequence = py_trees.composites.Sequence(name="Detect One Object", memory=True)
    detect_sequence.add_children([
        AskForPrompt(),
        MockGroundingDINO(),
        AskUserConfirmation(),
        ValidateDetection()
    ])

    # 4. The Loop Logic (Decorator)
    # If the sequence SUCCEEDS (we found 1 object), we convert it to RUNNING.
    # This forces the root Selector to run the loop again.
    # If the sequence FAILS (user said no), it returns FAILURE. 
    # The root Selector sees FAILURE and (because it's a Selector) tries the next child? 
    # No, we want it to retry THIS child.
    
    # Actually, simpler logic:
    # Use a generic Selector "Gather Loop".
    # Child 1: Check if done.
    # Child 2: Do work (Infinite Retry).
    
    # We wrap the detection sequence in a Retry decorator so if user says "No" (Failure), it immediately retries.
    retry_on_failure = py_trees.decorators.Retry(
        name="Retry Until Yes",
        child=detect_sequence,
        num_failures=999 # Retry infinitely on failure
    )

    # Now, if it Succeeds (User says Yes), we want to LOOP again.
    # We convert that Success to Running so the Tree keeps ticking.
    loop_forever = py_trees.decorators.SuccessIsRunning(
        name="LoopLogic",
        child=retry_on_failure
    )

    # 5. The Gather Selector
    # Priority: 1. Are we done? 2. Run the Loop
    gather_phase = py_trees.composites.Selector(name="Gather Loop", memory=False)
    gather_phase.add_children([check_done, loop_forever])

    # 6. The Finish Sequence
    finish_sequence = py_trees.composites.Sequence(name="Finish Phase", memory=False)
    finish_sequence.add_children([
        PlanTrajectory(),
        ExecuteTrajectory(),
    ])

    # 7. Root
    root = py_trees.composites.Sequence(name="Root", memory=True)
    root.add_children([gather_phase, finish_sequence])

    return py_trees.trees.BehaviourTree(root)

# ============================================================
# Runner
# ============================================================
def run():
    tree = create_tree()
    tree.setup(timeout=15)

    print("\n=== Behavior Tree Started ===")
    print("Goal: Detect 3 objects, then plan.")
    
    while True:
        try:
            tree.tick()
            
            # Check for global completion
            if tree.root.status == py_trees.common.Status.SUCCESS:
                print("\n✅ Tree finished successfully.")
                break
            if tree.root.status == py_trees.common.Status.FAILURE:
                print("❌ Tree failed.")
                break
                
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run()
