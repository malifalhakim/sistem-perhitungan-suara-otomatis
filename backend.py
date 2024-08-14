
from detection import get_bounding_boxes
from inference import get_texts
from crop import crop_image
import numpy as np

def process(img, CLIENT, ocr):
    # Get the bounding boxes
    bounding_boxes, inference_time = get_bounding_boxes(img, CLIENT)

    # Predict the texts
    result_dict = {}
    for no_urut in range(1,4):
        bounding_box = bounding_boxes[f"{no_urut}"]
        cropped_image = crop_image(img, bounding_box)
        text, confidence, ocr_time = get_texts(ocr, cropped_image)
        result_dict[no_urut] = {"text": text, "confidence": confidence, "ocr_time": ocr_time}

    result_dict['detect_time'] = inference_time

    return result_dict

def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image)
    # Konversi dari RGB (PIL) ke BGR (OpenCV)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image
