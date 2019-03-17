import imutils
import numpy as np
import cv2

def image_process_red(frame,lower_red1,lower_red2,upper_red1,upper_red2):
    
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, upper_red1, upper_red2)
    red_mask2 = cv2.inRange(hsv, lower_red1, lower_red2)
    mask_red = red_mask1 + red_mask2
    mask_red = cv2.erode(mask_red, None, iterations=2)
    mask_red = cv2.dilate(mask_red, None, iterations=2)
    
    return (mask_red)

def image_process(frame,lower,upper):
    
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    return (mask)