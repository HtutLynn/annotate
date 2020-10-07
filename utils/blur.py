import numpy as np
import cv2

def blur_bboxes(image, bboxes, blur=None):
    """
    Blur part of the image according to the bboxes for privacy

    Parameters
    ----------
    image  : cv2 image array
             Image, read by cv2.imread() function
    bboxes : numpy array
             [(x1, y1, x2, y2),..] format rectangle boxes
    blur   : str
             Blurring technique

    Return
    ------
    blurred_image : Image, whose rectangles ROIs are blurred
    """

    # If there is no bboxes, then return the image as it is
    if len(bboxes) == 0:
        return image
    else:
        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes)

        xywh_boxes = bboxes.copy()
        xywh_boxes[:, 2] = xywh_boxes[:, 2] - xywh_boxes[:, 0]
        xywh_boxes[:, 3] = xywh_boxes[:, 3] - xywh_boxes[:, 1]

        if blur is None: # the `blur` argument is None, go with typical blurring
            for box in xywh_boxes:
                # Grab ROI with Numpy slicing and blur
                x, y, w, h = box[:4].astype(np.int)
                roi = image[y:y+h, x:x+w]
                blurred_roi = cv2.blur(roi, (51, 51), 0)
                image[y:y+h, x:x+w] = blurred_roi

            return image

        elif blur == "Gaussian":
            for box in xywh_boxes:
                # Grab ROI with Numpy slicing and blur
                x, y, w, h = box[:4].astype(np.int)
                roi = image[y:y+h, x:x+w]
                blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                image[y:y+h, x:x+w] = blurred_roi

            return image

        elif blur == "Median":
            for box in xywh_boxes:
                # Grab ROI with Numpy slicing and blur
                x, y, w, h = box[:4].astype(np.int)
                roi = image[y:y+h, x:x+w]
                blurred_roi = cv2.medianBlur(roi,  51, 0)
                image[y:y+h, x:x+w] = blurred_roi

            return image

        elif blur == "Bilateral":
            for box in xywh_boxes:
                # Grab ROI with Numpy slicing and blur
                x, y, w, h = box[:4].astype(np.int)
                roi = image[y:y+h, x:x+w]
                blurred_roi = cv2.bilateralFilter(roi,9,75,75)
                image[y:y+h, x:x+w] = blurred_roi

            return image