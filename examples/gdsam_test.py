import cv2
import logging

from anypick_dk.grounded_sam_wrapper import GroundedSamWrapper

logging.basicConfig(level=logging.INFO)


def main():
    gdsam = GroundedSamWrapper()
    img = cv2.imread("../media/demo/demo4.jpg") 
    prompt = ["the brown dog"]
    bboxes, masks = gdsam.detect_and_segment(img, prompt)

    gdsam.annotate_and_save(img, output_path="grounded_sam_test.jpg")


if __name__ == "__main__":
    main()
