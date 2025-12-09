import importlib.resources as resources
import logging
import json
import numpy as np
import os
import subprocess
import tempfile

from pathlib import Path
from typing import List, Optional


class GraspPose:
    def __init__(self, position, orientation, score, width):
        self.position = np.array(position)
        self.orientation = np.array(orientation)
        self.score = float(score)
        self.width = float(width)

    def to_dict(self):
        return {
            'position': self.position.tolist(),
            'orientation': self.orientation.tolist(),
            'score': self.score,
            'width': self.width
        }

    def to_matrix(self):
        qw, qx, qy, qz = self.orientation
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.position
        return T

    def __repr__(self):
        return (f"GraspPose(pos={self.position}, "
                f"quat={self.orientation}, "
                f"score={self.score:.3f}, "
                f"width={self.width:.3f})")


class GraspDetector:

    def __init__(self, config_file: str):
        self.logger = logging.getLogger(__name__)
        self.exec_file = str(
            resources.files("anypick_dk")
            .joinpath("..", "..", "external", "anypick_gpd", "build", "detect_grasps_to_file")
            .resolve()
        )
        self.config_file = config_file

    def detect_grasps(self, pc_file: str, out_file: Optional[str] = None) -> List[GraspPose]:

        pc_path = Path(pc_file)
        if not pc_path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {pc_path}")

        config_path = Path(self.config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Create temporary output file for GPD
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            temp_output = tmp.name

        # Run GPD using custom C++ program
        grasps = self._run_gpd_and_parse(
            str(config_path), 
            str(pc_path), 
            temp_output
        )

        # Save to output file if not None
        if out_file is not None:
            self.save_grasps(grasps, Path(out_file))

        # Clean up temporary file
        if os.path.exists(temp_output):
            os.remove(temp_output)

        return grasps
   
    def _run_gpd_and_parse(self, config_file: str, pcd_file: str, output_file: str) -> List[GraspPose]:
        process = subprocess.Popen(
            [self.exec_file, config_file, pcd_file, output_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in process.stdout:
            self.logger.debug(line)

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"GPD exited with code {process.returncode}")

        if os.path.exists(output_file):
            grasps = self._parse_grasp_file(output_file)
            return grasps
        else:
            self.logger.warning(f"Warning: Output file not created: {output_file}")
            return []

    def _parse_grasp_file(self, filename: str) -> List[GraspPose]:
        grasps = []

        with open(filename, 'r') as f:
            lines = f.readlines()

        data_lines = [l.strip() for l in lines if l.strip() and not l.startswith('#')]

        if not data_lines:
            return grasps

        num_grasps = int(data_lines[0])
        for i in range(1, min(num_grasps + 1, len(data_lines))):
            values = list(map(float, data_lines[i].split()))
            grasp = GraspPose(
                position=values[0:3],
                orientation=values[3:7],
                score=values[7],
                width=values[8]
            )
            grasps.append(grasp)
        return grasps

    def save_grasps(self, grasps: List[GraspPose], filename: Path):
        json_file = filename.with_suffix('.json')
        with open(json_file, 'w') as f:
            json.dump([g.to_dict() for g in grasps], f, indent=4)

    def load_grasps(self, filename: str) -> List[GraspPose]:
        filepath = Path(filename)
        json_file = filepath.with_suffix('.json')
        with open(json_file, 'r') as f:
            data = json.load(f)
        return [GraspPose(**g) for g in data]