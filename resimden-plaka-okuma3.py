import cv2
import numpy as np
import pytesseract
import imutils

img = cv2.imread("licence_plate.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filtered = cv2.bilateralFilter(gray, 6, 250, 250)
edges = cv2.Canny(filtered, 30, 200)

contours,_ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


cv2.imshow("img", img)
cv2.imshow("gray", gray)
cv2.imshow("filtered", filtered)
cv2.imshow("edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()