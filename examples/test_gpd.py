import importlib.resources as resources
import logging

from anypick_dk.grasp_detector import GraspDetector

logging.basicConfig(level=logging.INFO)


def main():
    config_file = str(resources.files("anypick_dk") / "cfg" / "gpd_params.cfg")
    detector = GraspDetector(config_file)

    grasps = detector.detect_grasps("./obj_pc.pcd")

    # Print grasps sorted by score
    grasps_sorted = sorted(grasps, key=lambda g: g.score, reverse=True)
    for i, grasp in enumerate(grasps_sorted):
        print(f"\nGrasp {i+1}:")
        print(f"  Score:       {grasp.score:.4f}")
        print(f"  Position:    [{grasp.position[0]:.4f}, {grasp.position[1]:.4f}, {grasp.position[2]:.4f}] m")
        print(f"  Orientation: [{grasp.orientation[0]:.4f}, {grasp.orientation[1]:.4f}, "
              f"{grasp.orientation[2]:.4f}, {grasp.orientation[3]:.4f}]")
        print(f"  Width:       {grasp.width:.4f} m")
        print(f"  Transform matrix:")
        for row in grasp.to_matrix():
            print(f"    [{row[0]:7.4f}, {row[1]:7.4f}, {row[2]:7.4f}, {row[3]:7.4f}]")


if __name__ == "__main__":
    main()