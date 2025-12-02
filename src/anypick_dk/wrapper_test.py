from anypick_dk.grounded_sam_wrapper import GroundedSamWrapper
import cv2

def main():
    wrapper = GroundedSamWrapper()
    img = cv2.imread("../../media/demo/demo4.jpg") 
    prompt = ["the brown dog"]
    bboxes, masks = wrapper.detect_and_segment(img, prompt)

    wrapper.annotate_and_save(img, output_path="grounded_sam_all.jpg")


if __name__ == "__main__":
    main()
