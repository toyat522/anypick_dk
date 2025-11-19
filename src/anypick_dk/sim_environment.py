import logging
import numpy as np
import time

from anypick_dk.constants import (
    IIWA_LEN,
    IRIS_ANIM_INTERP,
    IRIS_ANIM_SLEEP,
)
from manipulation.meshcat_utils import PublishPositionTrajectory
from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import (
    CompositeTrajectory,
    DiagramBuilder,
    HPolyhedron,
    MathematicalProgram,
    Meshcat,
    MultibodyPlant,
    PiecewisePolynomial,
    Solve,
    StartMeshcat,
)


class SimEnvironment:
    def __init__(self, scenario_file: str):
        self.meshcat: Meshcat = StartMeshcat()
        scenario = LoadScenario(filename=scenario_file)
        builder = DiagramBuilder()

        self.station = builder.AddSystem(MakeHardwareStation(scenario, self.meshcat))
        self.plant: MultibodyPlant = self.station.GetSubsystemByName("plant")
        self.iiwa = self.plant.GetModelInstanceByName("iiwa")
        self.wsg = self.plant.GetModelInstanceByName("wsg")
        self.ee_frame = self.plant.GetFrameByName("body")
        self.visualizer = self.station.GetSubsystemByName("meshcat_visualizer(illustration)")

        self.diagram = builder.Build()

        self.diagram_context = self.diagram.CreateDefaultContext()
        self.station_context = self.diagram.GetSubsystemContext(self.station, self.diagram_context)
        self.plant_context = self.plant.GetMyContextFromRoot(self.diagram_context)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def get_iiwa_position(self) -> tuple:
        return self.plant.GetPositions(self.plant_context, self.iiwa)

    def set_iiwa_position(self, q: tuple) -> None:
        self.plant.SetPositions(self.plant_context, self.iiwa, q)

    def get_wsg_position(self) -> float:
        return np.abs(self.plant.GetPositions(self.plant_context, self.wsg)).mean()

    def set_wsg_position(self, setpoint: float) -> None:
        self.plant.SetPositions(self.plant_context, self.wsg, [-abs(setpoint) / 2, abs(setpoint) / 2])

    def publish_diagram(self) -> None:
        self.diagram.ForcedPublish(self.diagram_context)

    def animate_iris(self, region: HPolyhedron) -> None:
        q = region.ChebyshevCenter()[:IIWA_LEN]
        self.set_iiwa_position(q)
        self.publish_diagram()

        self.logger.info("Press the 'Stop Animation' button in Meshcat to continue")
        self.meshcat.AddButton("Stop Animation", "Escape")

        rng = np.random.default_rng()
        nq = self.plant.num_positions()
        prog = MathematicalProgram()
        qvar = prog.NewContinuousVariables(nq, "q")
        prog.AddLinearConstraint(region.A(), 0 * region.b() - np.inf, region.b(), qvar)
        cost = prog.AddLinearCost(np.ones((nq, 1)), qvar)

        while self.meshcat.GetButtonClicks("Stop Animation") < 1:
            direction = rng.standard_normal(nq)
            cost.evaluator().UpdateCoefficients(direction)
            result = Solve(prog)
            assert result.is_success()
            q_next = result.GetSolution(qvar)[:IIWA_LEN]
            for t in np.append(np.arange(0, 1, IRIS_ANIM_INTERP * np.linalg.norm(q_next - q)), 1):
                qs = t * q_next + (1 - t) * q
                self.set_iiwa_position(qs)
                self.publish_diagram()
                time.sleep(IRIS_ANIM_SLEEP)
            q = q_next

        self.meshcat.DeleteButton("Stop Animation")

    def visualize_traj(self, traj: CompositeTrajectory) -> None:
        # Sample the 9-DOF trajectory
        t_samples = np.linspace(traj.start_time(), traj.end_time(), 100)
        robot_positions = np.array([traj.value(t).flatten() for t in t_samples]).T  # 9 x N
        
        # Create full 51-DOF trajectory
        num_positions = self.plant.num_positions()
        full_positions = np.zeros((num_positions, len(t_samples)))
        
        # First 9 DOFs: robot trajectory from GCS
        full_positions[:9, :] = robot_positions
        
        # Remaining 42 DOFs: keep objects stationary at their current positions
        full_state = self.plant.GetPositions(self.plant.GetMyContextFromRoot(self.diagram_context))
        full_positions[9:, :] = full_state[9:].reshape(-1, 1)
        
        # Create the padded trajectory
        full_traj = PiecewisePolynomial.CubicShapePreserving(t_samples, full_positions)
        
        # Visualize trajectory
        PublishPositionTrajectory(full_traj, self.diagram_context, self.plant, self.visualizer)
        self.visualizer.ForcedPublish(self.visualizer.GetMyContextFromRoot(self.diagram_context))

    def clear_visualization(self) -> None:
        self.meshcat.Delete()
