# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import imutils
import numpy as np
import time
import cv2
 
lower_red1 = np.array([0,100,100])
lower_red2 = np.array([10,255,255])
upper_red1 = np.array([160,100,100])
upper_red2 = np.array([179,255,255])

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 60
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, upper_red1, upper_red2)
    red_mask2 = cv2.inRange(hsv, lower_red1, lower_red2)
    mask = red_mask1 + red_mask2
    res = cv2.bitwise_and(image,image, mask= mask)
    
    # show the frame
    cv2.imshow("Frame", image)
    cv2.imshow("Mask", mask)
    cv2.imshow("Overlay", res)
    key = cv2.waitKey(1) & 0xFF
 
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break