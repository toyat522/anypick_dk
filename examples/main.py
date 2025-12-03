import importlib.resources as resources
import logging

from anypick_dk.planner import Planner
from anypick_dk.sim_environment import SimEnvironment
from pydrake.all import GcsTrajectoryOptimization
Subgraph = GcsTrajectoryOptimization.Subgraph

logging.basicConfig(level=logging.INFO)


def main():
    scenario_file = str(resources.files("anypick_dk") / "scenarios" / "shelf_no_objects.yaml")
    sim_env = SimEnvironment(scenario_file)
    planner = Planner(sim_env)

if __name__ == "__main__":
    main()
