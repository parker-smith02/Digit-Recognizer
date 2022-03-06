import sys
import numpy as np
import cv2 as cv
import os
import sys


for image in os.listdir('temp'):
    if image.endswith('.jpg'):
        os.remove('temp/' + image)
for image in os.listdir('numbers'):
    if image.endswith('.jpg'):
        os.remove('numbers/' + image)


img = cv.imread('input.jpg')
blur = cv.GaussianBlur(img, (7, 7), 0)
gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

thresh = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)[1]
cv.imwrite('temp/test.jpg', thresh)

kern = cv.getStructuringElement(cv.MORPH_RECT, (3, 13))
dilate = cv.dilate(thresh, kern, iterations=5)  #not blurred enough, may need to adjust iterations

cnts = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
if len(cnts) == 2:
    cnts = cnts[0]
else: cnts = cnts[1]

cnts = sorted(cnts, key=lambda x: cv.boundingRect(x)[0])

index = 0
buffer = 20
for c in cnts:
    x, y, w, h = cv.boundingRect(c)

    roi = img[y - buffer:y+h + buffer, x - buffer:x+w + buffer]
    try:
        cv.imwrite('numbers/{}.jpg'.format(index), roi)

    except:
        continue

    cv.rectangle(img, (x, y), (x+w, y+h), (36, 255, 12), 2)
    
    index += 1

cv.imwrite('temp/bbox.jpg', img)

