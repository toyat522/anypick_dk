import importlib.resources as resources
import numpy as np

from anypick_dk.grasp_detector import GraspDetector


def main():
    config_file = str(resources.files("anypick_dk") / "cfg" / "gpd_params.cfg")
    detector = GraspDetector(config_file)
    
    # Detect grasps
    grasps = detector.detect_grasps("./obj_pc.pcd")
    
    # Sort by score
    grasps_sorted = sorted(grasps, key=lambda g: g.score, reverse=True)
    
    # Show top 5 grasps
    num_to_show = min(5, len(grasps_sorted))
    for i, grasp in enumerate(grasps_sorted[:num_to_show]):
        print(f"\nGrasp {i+1}:")
        print(f"  Score:       {grasp.score:.4f}")
        print(f"  Position:    [{grasp.position[0]:.4f}, {grasp.position[1]:.4f}, {grasp.position[2]:.4f}] m")
        print(f"  Orientation: [{grasp.orientation[0]:.4f}, {grasp.orientation[1]:.4f}, "
              f"{grasp.orientation[2]:.4f}, {grasp.orientation[3]:.4f}]")
        print(f"  Width:       {grasp.width:.4f} m ({grasp.width*1000:.1f} mm)")
        print(f"  Transform matrix:")
        for row in grasp.to_matrix():
            print(f"    [{row[0]:7.4f}, {row[1]:7.4f}, {row[2]:7.4f}, {row[3]:7.4f}]")


if __name__ == "__main__":
    main()