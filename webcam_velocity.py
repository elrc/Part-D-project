"""
Code to track and measure velocity of object using live webcam feed
Written by: Robert Crocker
Email contact: r.crocker-14@student.lboro.ac.uk
Extracts of code taken from:
https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
https://www.pyimagesearch.com/2016/01/04/unifying-picamera-and-cv2-videocapture-into-a-single-class-with-opencv/
Date created: 27/01/19
Last updated: 05/02/19
"""

# Imports all of the necessary packages for the code to work
from __future__ import print_function
from shapedetector import ShapeDetector
from videostream import VideoStream
import imutils
import numpy as np
import cv2
import time
import math
import argparse

def nothing(x):
    pass

# Passes the arguments given by the user when initalising the code
# into the code for use later
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--angle", type=float, default=0,
    help="River angle")
ap.add_argument("-d", "--distance", type=float, default=1,
    help="Distance from camera to ground")
args = vars(ap.parse_args())

# Starts the webcam video stream
cap = VideoStream(usePiCamera=False).start()
#cap.set(3,1920)
#cap.set(4,1080)

# Inialises the maths functions and values which are used later in
# calculations
pi = math.pi
tan = math.tan
cos = math.cos
sin = math.sin
sqrt = math.sqrt
atan = math.atan
hp = 960
vp = 540
#fr = 30
#t = 1/fr
d = args["distance"]
dd = d * 100
strd = "%dcm" % dd

# Sets the angle that the river is at 
strrivang = "%d degrees" % round(args["angle"])
riv_ang = args["angle"] * (pi / 180)
pt1x = 860
pt1y = 30
pt2x = round(pt1x+100*sin(riv_ang))
pt2y = round(pt1y+100*cos(riv_ang))

Dl = sqrt(9 ** 2 + 16 ** 2)
rat = atan(9 / 16)
fv = 78 * (pi / 180)
hfv = 2 * atan(tan(fv / 2) * (16 / Dl))
hhfv = hfv / 2
vfv = 2 * atan(tan(fv / 2) * (9 / Dl))
hvfv = vfv / 2
rpph = hfv / hp
rppv = vfv / vp
ang1 = fv / 2

i = 0
sv = []
av = []

strsmax = "N/A"
strsavg = "N/A"
strvmax = "N/A"
strvavg = "N/A"

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 10, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
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
    cv2.imshow("mask", mask)
    
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
    frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
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
    cv2.imshow("mask", mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()

while(True):
    # Capture frame-by-frame
    frame = cap.read()
    if i != 0:
        end_timer = time.time()
        t = end_timer - start_timer
    start_timer = time.time()
    
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, upper_red1, upper_red2)
    red_mask2 = cv2.inRange(hsv, lower_red1, lower_red2)
    mask = red_mask1 + red_mask2
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    sd = ShapeDetector()

    # loop over the contours
    for c in cnts:
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            M = cv2.moments(c)
            if M['m00']>100:
                cX = int((M['m10'] / M['m00']))
                cY = int((M['m01'] / M['m00']))
                if cX > 240 and cX < 1680:
                    if cY > 135 and cY < 945:
                        shape = sd.detect(c)
                        if shape == "square":
                            if i == 0:
                                pcX = cX
                                pcY = cY
                                strs = "0"
                                strsmax = "N/A"
                                strsavg = "N/A"
                                strvmax = "N/A"
                                strvavg = "N/A"
                                i = 1
                            else:
                                w = 2 * d * tan(ang1)
                                x = w * cos(rat)
                                y = w * sin(rat)
                                prcx = rpph * pcX
                                prcy = rppv * pcY
                                rcx= rpph * cX
                                rcy = rppv * cY
                                
                                if prcx <= hhfv:
                                    x1 = (x / 2) - (d * tan(hhfv - prcx))
                                elif prcx > hhfv:
                                    x1 = (x / 2) + (d * tan(prcx - hhfv))
                                if prcy <= hvfv:
                                    y1 = (y / 2) - (d * tan(hvfv - prcy))
                                elif prcy > hvfv:
                                    y1 = (y / 2) + (d * tan(prcy - hvfv))
                                
                                if rcx <= hhfv:
                                    x2 = (x / 2) - (d * tan(hhfv - rcx))
                                elif rcx > hhfv:
                                    x2 = (x / 2) + (d * tan(rcx - hhfv))
                                if rcy <= hvfv:
                                    y2 = (y / 2) - (d * tan(hvfv - rcy))
                                elif rcy > hvfv:
                                    y2 = (y / 2) + (d * tan(rcy - hvfv))

                                ds = sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
                                if x1 == x2:
                                    vel_ang = 0
                                elif y1 == y2:
                                    vel_ang = pi / 2
                                else:
                                    vel_ang = atan((x1 - x2) / (y1 - y2))
                                vel_ang_deg = vel_ang * (180 / pi)
                                s = ds / t
                                strs = "%f m/s" % s
                                sv.append(s)
                                smax = max(sv)
                                savg = sum(sv) / len(sv)
                                strsmax = "Max Speed = %f m/s" % smax
                                strsavg = "Average Speed = %f m/s" % savg
                                ang_vel = s * cos(riv_ang - vel_ang)
                                av.append(ang_vel)
                                vmax = max(av)
                                vavg = sum(av) / len(av)
                                strvmax = "Max Velocity = %f m/s" % vmax
                                strvavg = "Average Velocity = %f m/s" % vavg
                                pcX = cX
                                pcY = cY
                            # multiply the contour (x, y)-coordinates by the resize ratio,
                            # then draw the contours and the name of the shape on the image
                            c = c.astype("int")
                            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                            cv2.putText(frame, strs, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, strsmax, (1, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 0, 0), 2)
    cv2.putText(frame, strsavg, (1, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 0, 0), 2)
    cv2.putText(frame, strvmax, (1, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 0, 0), 2)
    cv2.putText(frame, strvavg, (1, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 0, 0), 2)
    cv2.putText(frame, strd, (1, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 0, 0), 2)
    cv2.putText(frame, strrivang, (pt1x-15, pt1y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 0, 255), 2)
    cv2.arrowedLine(frame, (pt1x, pt1y), (pt2x, pt2y), (0, 0, 255), 3)
    
    # show the frame
    cv2.imshow('frame',frame)
    #cv2.imshow("Mask", mask)
    #cv2.imshow("Overlay", res)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.stop()
cv2.destroyAllWindows()
    