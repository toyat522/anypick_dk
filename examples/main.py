import importlib.resources as resources
import numpy as np

from anypick_dk.constants import (
    IIWA_LEN, q_BotShelfPlace, q_MidShelfPlace, q_Object, q_TopShelfPlace, WSG_LEN, WSG_VEL_BOUND
)
from anypick_dk.planner import Planner
from anypick_dk.sim_environment import SimEnvironment
from pydrake.all import GcsTrajectoryOptimization, HPolyhedron, LoadIrisRegionsYamlFile, Point
Subgraph = GcsTrajectoryOptimization.Subgraph


def create_base_gcs(iris_regions: dict[HPolyhedron],
                    sim_env: SimEnvironment) -> tuple[GcsTrajectoryOptimization, dict[Subgraph]]:

    gcs = GcsTrajectoryOptimization(IIWA_LEN + WSG_LEN)

    nodes = {
        "start": gcs.AddRegions(
            [iris_regions["start_region"]], order=1, name="start"
        ),
        "home": gcs.AddRegions(
            [iris_regions["home_region"]], order=1, name="home"
        ),
        "transition": gcs.AddRegions(
            [iris_regions["transition_region"]], order=1, name="transition"
        ),
        "top_shelf": gcs.AddRegions(
            [iris_regions["top_shelf_region"]], order=1, name="top_shelf"
        ),
        "top_shelf_approach": gcs.AddRegions(
            [iris_regions["top_shelf_approach_region"]], order=1, name="top_shelf_approach"
        ),
        "mid_shelf": gcs.AddRegions(
            [iris_regions["mid_shelf_region"]], order=1, name="mid_shelf"
        ),
        "mid_shelf_approach": gcs.AddRegions(
            [iris_regions["mid_shelf_approach_region"]], order=1, name="mid_shelf_approach"
        ),
        "bot_shelf": gcs.AddRegions(
            [iris_regions["bot_shelf_region"]], order=1, name="bot_shelf_region"
        ),
        "bot_shelf_approach": gcs.AddRegions(
            [iris_regions["bot_shelf_approach_region"]], order=1, name="bot_shelf_approach"
        ),
        "object": gcs.AddRegions(
            [iris_regions["object_region"]], order=1, name="object"
        ),
        "top_shelf_place": gcs.AddRegions(
            [Point(np.concat([q_TopShelfPlace, np.zeros(WSG_LEN)]))], order=0, name="top_shelf_place"
        ),
        "mid_shelf_place": gcs.AddRegions(
            [Point(np.concat([q_MidShelfPlace, np.zeros(WSG_LEN)]))], order=0, name="mid_shelf_place"
        ),
        "bot_shelf_place": gcs.AddRegions(
            [Point(np.concat([q_BotShelfPlace, np.zeros(WSG_LEN)]))], order=0, name="bot_shelf_place"
        )
    }

    gcs.AddEdges(nodes["start"], nodes["home"])

    gcs.AddEdges(nodes["home"], nodes["object"])
    gcs.AddEdges(nodes["object"], nodes["home"])

    gcs.AddEdges(nodes["home"], nodes["top_shelf_approach"])
    gcs.AddEdges(nodes["top_shelf_approach"], nodes["home"])

    gcs.AddEdges(nodes["home"], nodes["mid_shelf_approach"])
    gcs.AddEdges(nodes["mid_shelf_approach"], nodes["home"])

    gcs.AddEdges(nodes["home"], nodes["transition"])
    gcs.AddEdges(nodes["transition"], nodes["home"])

    gcs.AddEdges(nodes["transition"], nodes["bot_shelf_approach"])
    gcs.AddEdges(nodes["bot_shelf_approach"], nodes["transition"])

    gcs.AddEdges(nodes["top_shelf_approach"], nodes["top_shelf"])
    gcs.AddEdges(nodes["top_shelf"], nodes["top_shelf_approach"])

    gcs.AddEdges(nodes["mid_shelf_approach"], nodes["mid_shelf"])
    gcs.AddEdges(nodes["mid_shelf"], nodes["mid_shelf_approach"])

    gcs.AddEdges(nodes["bot_shelf_approach"], nodes["bot_shelf"])
    gcs.AddEdges(nodes["bot_shelf"], nodes["bot_shelf_approach"])

    gcs.AddEdges(nodes["top_shelf"], nodes["top_shelf_place"])
    gcs.AddEdges(nodes["top_shelf_place"], nodes["top_shelf"])

    gcs.AddEdges(nodes["mid_shelf"], nodes["mid_shelf_place"])
    gcs.AddEdges(nodes["mid_shelf_place"], nodes["mid_shelf"])

    gcs.AddEdges(nodes["bot_shelf"], nodes["bot_shelf_place"])
    gcs.AddEdges(nodes["bot_shelf_place"], nodes["bot_shelf"])

    gcs.AddTimeCost()
    lb = sim_env.plant.GetVelocityLowerLimits()[:IIWA_LEN + WSG_LEN]
    lb[~np.isfinite(lb)] = -WSG_VEL_BOUND
    ub = sim_env.plant.GetVelocityUpperLimits()[:IIWA_LEN + WSG_LEN]
    ub[~np.isfinite(ub)] = WSG_VEL_BOUND
    gcs.AddVelocityBounds(lb, ub)
    return gcs, nodes


def main():
    scenario_file = str(resources.files("anypick_dk") / "scenarios" / "shelf_no_objects.yaml")
    sim_env = SimEnvironment(scenario_file)
    planner = Planner(sim_env)

    iris_regions_file = str(resources.files("anypick_dk") / "iris_regions" / "shelf_regions.yaml")
    iris_regions = LoadIrisRegionsYamlFile(iris_regions_file)
    gcs, nodes = create_base_gcs(iris_regions, sim_env)


if __name__ == "__main__":
    main()

