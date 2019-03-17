from __future__ import print_function
from videostream import VideoStream
import numpy as np
import imutils
import time
import cv2

def nothing(x):
    pass

vs = VideoStream(usePiCamera=False).start()

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 10, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    frame = vs.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    value = cv2.split(hsv)[2]
    average1,_,_,_ = cv2.mean(value)
    
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    lower_red1 = np.array([l_h, l_s, l_v])
    lower_red2 = np.array([u_h, u_s, u_v])
    
    mask = cv2.inRange(hsv, lower_red1, lower_red2)
    
    cv2.imshow("frame", frame)
    cv2.imshow("lower red", mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 160, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    frame = vs.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    value = cv2.split(hsv)[2]
    average2,_,_,_ = cv2.mean(value)
    
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    upper_red1 = np.array([l_h, l_s, l_v])
    upper_red2 = np.array([u_h, u_s, u_v])
    
    mask = cv2.inRange(hsv, upper_red1, upper_red2)
    
    cv2.imshow("frame", frame)
    cv2.imshow("upper red", mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()

while True:
    frame = vs.read()
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = cv2.split(image)[2]
    average,_,_,_ = cv2.mean(value)
    
    if average != average1:
        diff1 = average - average1
        lower_red1[2] = lower_red1[2] + round(diff1)
        average1 = average
    if average != average2:
        diff2 = average - average2
        upper_red1[2] = upper_red1[2] + round(diff2)
        average2 = average
    
    print(lower_red1[2], diff1, upper_red1[2], diff2)
    
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, upper_red1, upper_red2)
    red_mask2 = cv2.inRange(hsv, lower_red1, lower_red2)
    mask = red_mask1 + red_mask2
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    #image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #value = cv2.split(image)[2]
    #average,_,_,_ = cv2.mean(value)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #avg,_,_,_ = cv2.mean(gray)
    #print(avg, average)
    #print(average)
    #cv2.imshow('Frame', frame)
    #cv2.imshow('Value', value)
    #cv2.imshow('Gray', gray)
    
    cv2.imshow('mask', mask)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()