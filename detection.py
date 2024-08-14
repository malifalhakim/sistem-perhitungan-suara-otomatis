import cv2

def get_bounding_boxes(img, CLIENT):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = CLIENT.infer(gray_image, model_id="pemilu-gemastik/1")
    
    bounding_boxes = result['predictions']
    inference_time = result['time']

    dict_bounding_boxes = {}
    for bounding_boxes in bounding_boxes:
        x1 = bounding_boxes['x'] - bounding_boxes['width'] / 2
        x2 = bounding_boxes['x'] + bounding_boxes['width'] / 2
        y1 = bounding_boxes['y'] - bounding_boxes['height'] / 2
        y2 = bounding_boxes['y'] + bounding_boxes['height'] / 2
        class_name = bounding_boxes['class']
        box = (x1, y1, x2, y2)
        dict_bounding_boxes[class_name] = box
    return dict_bounding_boxes, inference_time
