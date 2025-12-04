import logging
import numpy as np
import time

from anypick_dk.constants import IIWA_LEN, SIM_END_SECS
from anypick_dk.utils import reshape_trajectory
from manipulation.meshcat_utils import PublishPositionTrajectory
from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import (
    CameraInfo,
    CompositeTrajectory,
    DiagramBuilder,
    HPolyhedron,
    MathematicalProgram,
    Meshcat,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    RevoluteJoint,
    RigidTransform,
    Simulator,
    Solve,
    StartMeshcat,
    TrajectorySource,
)
from typing import Optional


class SimEnvironment:

    sim_time: Optional[float] = None
    plant: MultibodyPlant

    def __init__(self, scenario_file: str):
        self.logger = logging.getLogger(__name__)

        self.meshcat: Meshcat = StartMeshcat()
        self.scenario = LoadScenario(filename=scenario_file)

        builder = DiagramBuilder()
        self._build_default_diagram(builder)

        self.diagram = builder.Build()
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.station_context = self.diagram.GetSubsystemContext(self.station, self.diagram_context)
        self.plant_context = self.plant.GetMyContextFromRoot(self.diagram_context)
        self.publish_diagram()

    def build_diagram_with_controller(self, iiwa_traj: CompositeTrajectory,
                                      wsg_traj: CompositeTrajectory) -> None:
        iiwa_traj = reshape_trajectory(iiwa_traj, out_dim=IIWA_LEN)
        self.sim_time = iiwa_traj.end_time() + SIM_END_SECS

        builder = DiagramBuilder()
        self._build_default_diagram(builder)

        controller_plant = MultibodyPlant(time_step=self.plant.time_step())
        self._add_iiwa(controller_plant)
        controller_plant.Finalize()

        iiwa_traj_source = builder.AddSystem(TrajectorySource(iiwa_traj))
        builder.Connect(
            iiwa_traj_source.get_output_port(),
            self.station.GetInputPort("iiwa.position")
        )

        wsg_traj_source = builder.AddSystem(TrajectorySource(wsg_traj))
        builder.Connect(
            wsg_traj_source.get_output_port(),
            self.station.GetInputPort("wsg.position")
        )

        self.diagram = builder.Build()
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.station_context = self.diagram.GetSubsystemContext(self.station, self.diagram_context)
        self.plant_context = self.plant.GetMyContextFromRoot(self.diagram_context)

    def _build_default_diagram(self, builder: DiagramBuilder) -> None:
        self.station = builder.AddSystem(MakeHardwareStation(self.scenario, self.meshcat))
        self.plant = self.station.GetSubsystemByName("plant")
        self.iiwa = self.plant.GetModelInstanceByName("iiwa")
        self.wsg = self.plant.GetModelInstanceByName("wsg")
        self.ee_frame = self.plant.GetFrameByName("body")
        self.visualizer = self.station.GetSubsystemByName("meshcat_visualizer(illustration)")

    def _add_iiwa(self, plant: MultibodyPlant) -> ModelInstanceIndex:
        parser = Parser(plant)
        iiwa = parser.AddModelsFromUrl(
            f"package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf"
        )[0]
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
        q0 = [0.0, 0.1, 0, -1.2, 0, 1.6, 0]
        index = 0
        for joint_index in plant.GetJointIndices(iiwa):
            joint = plant.get_mutable_joint(joint_index)
            if isinstance(joint, RevoluteJoint):
                joint.set_default_angle(q0[index])
                index += 1
        return iiwa
    
    def get_iiwa_position(self) -> tuple:
        return self.plant.GetPositions(self.plant_context, self.iiwa)

    def set_iiwa_position(self, q: tuple) -> None:
        self.plant.SetPositions(self.plant_context, self.iiwa, q)

    def get_wsg_position(self) -> float:
        return np.abs(self.plant.GetPositions(self.plant_context, self.wsg)).mean()

    def set_wsg_position(self, setpoint: float) -> None:
        self.plant.SetPositions(self.plant_context, self.wsg, [-abs(setpoint) / 2, abs(setpoint) / 2])
    
    def get_camera_bgr(self, idx: int) -> np.ndarray:
        img = self.station.GetOutputPort(f"camera{idx}.rgb_image").Eval(self.station_context)
        img = np.array(img.data, copy=False).reshape(img.height(), img.width(), -1)[:,:,:-1]
        return img[:,:,::-1]

    def get_camera_depth(self, idx: int) -> np.ndarray:
        img = self.station.GetOutputPort(f"camera{idx}.depth_image").Eval(self.station_context)
        depth_img = np.array(img.data, copy=False).reshape(img.height(), img.width(), -1)[:,:,0]
        return np.ma.masked_invalid(depth_img)

    def get_camera_pose(self, idx: int) -> RigidTransform:
        camera = self.station.GetSubsystemByName(f"rgbd_sensor_camera{idx}")
        camera_context = camera.GetMyContextFromRoot(self.diagram_context)
        return camera.body_pose_in_world_output_port().Eval(camera_context)

    def get_camera_intrinsics(self, idx: int) -> CameraInfo:
        camera = self.station.GetSubsystemByName(f"rgbd_sensor_camera{idx}")
        camera_context = camera.GetMyContextFromRoot(self.diagram_context)
        return camera.GetDepthRenderCamera(camera_context).core().intrinsics()

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
            for t in np.append(np.arange(0, 1, 20 * np.linalg.norm(q_next - q)), 1):
                qs = t * q_next + (1 - t) * q
                self.set_iiwa_position(qs)
                self.publish_diagram()
                time.sleep(0.05)
            q = q_next

        self.meshcat.DeleteButton("Stop Animation")

    def visualize_traj(self, traj: CompositeTrajectory) -> None:
        full_state = self.plant.GetPositions(self.plant_context)
        full_traj = reshape_trajectory(
            traj,
            out_dim=self.plant.num_positions(),
            fill_value=full_state
        )
        PublishPositionTrajectory(full_traj, self.diagram_context, self.plant, self.visualizer)
        self.visualizer.ForcedPublish(self.visualizer.GetMyContextFromRoot(self.diagram_context))

    def clear_visualization(self) -> None:
        self.meshcat.Delete()

    def simulate(self) -> None:
        assert self.sim_time is not None, "SimEnvironment needs to call build_diagram_with_controller first"

        simulator = Simulator(self.diagram, self.diagram_context)
        self.logger.info(f"Simulation will run for {self.sim_time} seconds")

        self.meshcat.StartRecording()
        simulator.AdvanceTo(self.sim_time)
        self.meshcat.StopRecording()
        self.meshcat.PublishRecording()