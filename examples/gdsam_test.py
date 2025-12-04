import cv2

from anypick_dk.grounded_sam_wrapper import GroundedSamWrapper


def main():
    gdsam = GroundedSamWrapper()
    img = cv2.imread("../media/demo/demo4.jpg") 
    prompt = ["the brown dog"]
    gdsam.detect_and_segment(img, prompt, box_threshold=0.2)
    annotated = gdsam.annotate()
    cv2.imwrite("../media/demo/gdsam_test.jpg", annotated)


if __name__ == "__main__":
    main()
