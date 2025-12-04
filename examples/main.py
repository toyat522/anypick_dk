import importlib.resources as resources
import logging
import py_trees

from anypick_dk.orchestrator import Orchestrator
from anypick_dk.planner import Planner
from anypick_dk.sim_environment import SimEnvironment
from pydrake.all import GcsTrajectoryOptimization
Subgraph = GcsTrajectoryOptimization.Subgraph

logging.basicConfig(level=logging.INFO)


def main():
    logger = logging.getLogger(__name__)
    scenario_file = str(resources.files("anypick_dk") / "scenarios" / "shelf_with_objects.yaml")
    sim_env = SimEnvironment(scenario_file)
    planner = Planner(sim_env)
    orchestrator = Orchestrator()

    while True:
        result = orchestrator.tick()

        if result == py_trees.common.Status.SUCCESS:
            logger.info("Tree successfully finished!")
            break
        elif result == py_trees.common.Status.FAILURE:
            logger.warning("Tree failed!")


if __name__ == "__main__":
    main()
