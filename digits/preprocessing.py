import cv2
import numpy as np


def preprocess(img_path):
    '''
    Read the input image and crop the date and time areas. And normalize the cropped area.
    
    Args :
    img_path : path to the image

    Return :
    datetime_list : a python list containing the preprocessed areas (numpy array)
    '''
    # Read image in grayscale.
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    datetime_list = []
    
    # Cropping the date and time areas and store them in the list.
    month1 = img[56:97, 43:72, :].copy()
    datetime_list.append(month1)

    month2 = img[56:97, 71:95, :].copy()
    datetime_list.append(month2)

    day1 = img[56:97, 120:143, :].copy()
    datetime_list.append(day1)

    day2 = img[56:97, 141:167, :].copy()
    datetime_list.append(day2)

    year1 = img[56:97, 191:215, :].copy()
    datetime_list.append(year1)

    year2 = img[56:97, 214:240, :].copy()
    datetime_list.append(year2)

    year3 = img[56:97, 238:264, :].copy()
    datetime_list.append(year3)

    year4 = img[56:97, 262:289, :].copy()
    datetime_list.append(year4)

    hour1 = img[56:97, 405:434, :].copy()
    datetime_list.append(hour1)

    hour2 = img[56:97, 430:458, :].copy()
    datetime_list.append(hour2)

    minute1 = img[56:97, 476:504, :].copy()
    datetime_list.append(minute1)

    minute2 = img[56:97, 502:529, :].copy()
    datetime_list.append(minute2)

    second1 = img[56:97, 548:576, :].copy()
    datetime_list.append(second1)

    second2 = img[56:97, 573:603, :].copy()
    datetime_list.append(second2)

    dsize = (28, 28) # Dimensions which model expects
    for i in range(len(datetime_list)):
        image = datetime_list[i]

        # Resize to desired dimensions
        image = cv2.resize(image, dsize)
        image = image.astype(np.float32)

        # Normalization
        datetime_list[i] = (image/255 - 0.1307) / 0.3081

    return datetime_list
