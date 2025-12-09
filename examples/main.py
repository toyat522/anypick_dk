import importlib.resources as resources
import logging
import py_trees

from anypick_dk.constants import X_WCan, X_WGelatin, X_WMustard, X_WSugar
from anypick_dk.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#gt_poses = [X_WCan, X_WGelatin, X_WSugar]  # Set to None if using perception system
gt_poses = None


def main():
    scenario_file = str(resources.files("anypick_dk") / "scenarios" / "shelf_with_objects.yaml")
    gpd_config_file = str(resources.files("anypick_dk") / "cfg" / "gpd_params.cfg")

    orchestrator = Orchestrator(scenario_file, gpd_config_file, gt_poses)
    py_trees.display.render_dot_tree(orchestrator.tree.root)

    while True:
        result = orchestrator.tick()

        if result == py_trees.common.Status.SUCCESS:
            logger.info("Tree successfully finished!")
            break
        elif result == py_trees.common.Status.FAILURE:
            logger.error("Tree failed!")
            break


if __name__ == "__main__":
    main()
