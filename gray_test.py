from __future__ import print_function
from videostream import VideoStream
import numpy as np
import imutils
import time
import cv2

vs = VideoStream(usePiCamera=False).start()

while True:
    frame = vs.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = cv2.split(image)[2]
    average,_,_,_ = cv2.mean(value)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg,_,_,_ = cv2.mean(gray)
    print(avg, average)
    #print(average)
    cv2.imshow('Frame', frame)
    cv2.imshow('Value', value)
    cv2.imshow('Gray', gray)
    
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()