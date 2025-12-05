import importlib.resources as resources
import logging
import py_trees

from anypick_dk.constants import X_WCan, X_WGelatin, X_WMustard, X_WSugar
from anypick_dk.orchestrator import Orchestrator
from pydrake.all import GcsTrajectoryOptimization
Subgraph = GcsTrajectoryOptimization.Subgraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gt_poses = [X_WCan, X_WGelatin, X_WSugar]  # Set to None if using perception system


def main():
    scenario_file = str(resources.files("anypick_dk") / "scenarios" / "shelf_with_objects.yaml")
    orchestrator = Orchestrator(scenario_file, gt_poses)
    py_trees.display.render_dot_tree(orchestrator.tree.root)

    while True:
        result = orchestrator.tick()

        if result == py_trees.common.Status.SUCCESS:
            logger.info("Tree successfully finished!")
            break
        elif result == py_trees.common.Status.FAILURE:
            logger.warning("Tree failed!")


if __name__ == "__main__":
    main()
