import time

def get_texts(ocr_model, img):
    start = time.time()
    result = ocr_model.ocr(img, cls=False,det=False)
    end = time.time()

    return result[0][0][0], result[0][0][1], end-start