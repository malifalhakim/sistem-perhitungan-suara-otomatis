import cv2
import numpy as np

def crop_image(img, bounding_boxes):
    # Bird view cropping
    xmin, ymin, xmax, ymax = bounding_boxes
    points = [(xmin, ymax), (xmin, ymin), (xmax, ymin), (xmax, ymax)]
    pts1 = np.float32([[points[0][0], points[0][1]], [points[1][0], points[1][1]],
                          [points[2][0], points[2][1]], [points[3][0], points[3][1]]])
    
    width = 200
    height = 400

    pts2 = np.float32([[0, height], [0, 0], [width, 0], [width, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_output = cv2.warpPerspective(img, matrix, (width, height))

    return img_output