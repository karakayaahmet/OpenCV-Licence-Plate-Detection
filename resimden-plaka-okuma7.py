import cv2
from cv2 import bitwise_and
import numpy as np
import pytesseract
import imutils

img = cv2.imread("licence_plate.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filtered = cv2.bilateralFilter(gray, 6, 250, 250)
edges = cv2.Canny(filtered, 30, 200)

contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(contours)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
screen = None

for c in cnts:
    epsilon = 0.018*cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    if len(approx) == 4:
        screen = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(mask, [screen], 0, (255,255,255), -1)
new_img = cv2.bitwise_and(img, img, mask=mask)

(x,y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
cropped = gray[topx:bottomx+1, topy:bottomy+1]

text = pytesseract.image_to_string(cropped, lang="tur")
cv2.putText(mask, text, (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 3)

cv2.imshow("img", img)
cv2.imshow("gray", gray)
cv2.imshow("filtered", filtered)
cv2.imshow("edges", edges)
cv2.imshow("new_img", new_img)
cv2.imshow("cropped", cropped)
cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()