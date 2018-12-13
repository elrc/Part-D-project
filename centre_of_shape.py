# import the necessary packages
import imutils
import numpy as np
import cv2
 
 
# load the image, convert it to grayscale, blur it slightly,
# and threshold it
image = cv2.imread('shapes_and_colours.jpg',1)
cv2.imshow("Original Image", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Grey", gray)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#cv2.imshow("Blurred", blurred)
thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Thresh", thresh)

# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
# loop over the contours
for c in cnts:
    # compute the center of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
 
    # draw the contour and center of the shape on the image
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
 
    # show the image
cv2.imshow("Image", image)
cv2.waitKey(0)