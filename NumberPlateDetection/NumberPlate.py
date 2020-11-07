import cv2
import numpy as np

filename = "car8"
img = cv2.imread(filename+'.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

blur = cv2.bilateralFilter(gray,11,90,90)

edges = cv2.Canny(blur,30,200)

cnts, new = cv2.findContours(edges.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

image_copy = img.copy()

cnts = sorted(cnts, key=cv2.contourArea,reverse=True)[:30]

cv2.drawContours(image_copy,cnts,-1,(255,0,255),2)
plate = None

for c in cnts:
    perimeter = cv2.arcLength(c,True)
    edges_count = cv2.approxPolyDP(c,0.02*perimeter,True)
    if len(edges_count) == 4:
        x,y,w,h = cv2.boundingRect(c)
        plate = img[y:y+h,x:x+w]
        break
cv2.imwrite(filename+"result.jpg",plate)
