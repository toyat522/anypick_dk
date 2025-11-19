from anypick_dk.constants import IIWA_LEN
from pydrake.all import (
    AddDefaultVisualization,
    CompositeTrajectory,
    Context,
    Diagram,
    GcsTrajectoryOptimization,
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    HPolyhedron,
    InverseKinematics,
    IrisInConfigurationSpace,
    IrisOptions,
    LoadIrisRegionsYamlFile,
    MathematicalProgram,
    MultibodyPlant,
    Point,
    Rgba,
    RigidBodyFrame,
    RigidTransform,
    SaveIrisRegionsYamlFile,
    Solve,
    Sphere,
)
from manipulation.meshcat_utils import PublishPositionTrajectory
from manipulation.scenarios import AddIiwa, AddWsg
from manipulation.utils import ConfigureParser


def create_base_gcs() -> GcsTrajectoryOptimization:
    gcs = GcsTrajectoryOptimization(IIWA_LEN)

    home_node = gcs.AddRegions([home_region], order=1)
    start_node = gcs.AddRegions([start_region], order=1)
    shelf_node = gcs.AddRegions([shelf_region], order=1)

    source = gcs.AddRegions([Point(q_start)], order=0)
    target = gcs.AddRegions([Point(q_goal)], order=0)

    gcs.AddEdges(source, home_node)
    gcs.AddEdges(home_node, start_node)
    gcs.AddEdges(start_node, shelf_node)
    gcs.AddEdges(shelf_node, target)
    gcs.AddEdges(source, start_node)
    gcs.AddEdges(home_node, shelf_node)

    self.gcs.AddTimeCost()
    self.gcs.AddVelocityBounds(
        self.sim_env.plant.GetVelocityLowerLimits(),
        self.sim_env.plant.GetVelocityUpperLimits()
    )


def main():
    pass


if __name__ == "__main__":
    main()

