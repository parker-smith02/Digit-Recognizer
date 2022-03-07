import numpy as np
import cv2 as cv
import os
from PIL import Image

def crop_images():
    for image in os.listdir('temp'):
        if image.endswith('.jpg'):
            os.remove('temp/' + image)
    for image in os.listdir('numbers'):
        if image.endswith('.jpg'):
            os.remove('numbers/' + image)
 

    img = cv.imread('input2.jpg')
    blur = cv.GaussianBlur(img, (7, 7), 0)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

    thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)[1]
    cv.imwrite('temp/test.jpg', thresh)

    kern = cv.getStructuringElement(cv.MORPH_RECT, (3, 13))
    dilate = cv.dilate(thresh, kern, iterations=5)  #not blurred enough, may need to adjust iterations

    cv.imwrite('temp/dialate.jpg', dilate)
    cnts = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 2:
        cnts = cnts[0]
    else: cnts = cnts[1]

    cnts = sorted(cnts, key=lambda x: cv.boundingRect(x)[0])

    index = 0
    buffer = 20
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)

        roi = img[y-buffer:y+h, x-buffer:x+w+buffer]
        try:
            cv.imwrite('numbers/{}.jpg'.format(index), roi)

        except:
            continue
  
        if w < 10 or h < 10: continue
        if w > 1500 or h > 1500: continue


        cv.rectangle(img, (x, y), (x+w, y+h), (36, 255, 12), 2)
        
        index += 1

    cv.imwrite('temp/bbox.jpg', img)


def format_numbers():
    for file in os.listdir('numbers'):
        if file.endswith('.jpg'):
            image = cv.imread('numbers/{}'.format(file))
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            inv = cv.bitwise_not(gray)
            cv.imwrite('temp/inv_{}'.format(file), inv)
            thresh = cv.threshold(inv, 150, 255, cv.THRESH_BINARY)[1]
            descale = resize(thresh, 28, 28)
            cv.imwrite('numbers/{}'.format(file), descale)

def resize(img, height, width):
    dim = (width, height)
    return cv.resize(img, dim)

def image_to_np_array(filename):
    img = Image.open('numbers/{}'.format(filename))
    return np.asarray(img)



    
crop_images()
format_numbers()