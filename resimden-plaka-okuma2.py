import cv2
import numpy as np
import pytesseract
import imutils

img = cv2.imread("licence_plate.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filtered = cv2.bilateralFilter(gray, 5, 250, 250)


cv2.imshow("img", img)
cv2.imshow("gray", gray)
cv2.imshow("filtered", filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()