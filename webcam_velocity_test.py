"""
Code to track and measure velocity of object using live webcam feed
Written by: Robert Crocker
Email contact: r.crocker-14@student.lboro.ac.uk
Extracts of code taken from:
https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
https://www.pyimagesearch.com/2016/01/04/unifying-picamera-and-cv2-videocapture-into-a-single-class-with-opencv/
Date created: 27/01/19
Last updated: 17/03/19
"""

# Imports all of the necessary packages for the code to work
from __future__ import print_function
from shapedetector import ShapeDetector
from videostream import VideoStream
from maths import velocity_math
import colour_detect
import image_processing
import imutils
import numpy as np
import cv2
import time
import math
import argparse

river_angle = np.genfromtxt('/home/pi/PartD_Project/Variables/river_angle.csv')
river_angle.tolist()

# Passes the arguments given by the user when initalising the code
# into the code for use later
ap = argparse.ArgumentParser()
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
strrivang = "%d degrees" % river_angle
riv_ang = river_angle * (pi / 180)
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

i,h,t,p,pcX,pcY,ptcX,ptcY,smax,savg,vmax,vavg,ssmax,ssavg,vsmax,vsavg,stmax,stavg,vtmax,vtavg = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
sv,av,tv,atv = [],[],[],[]

strsavg = "N/A"
strvmax = "N/A"
strvavg = "N/A"

lower_red1,lower_red2,average1 = colour_detect.lower_red(cap)
upper_red1,upper_red2,average2 = colour_detect.upper_red(cap)

while(True):
    # Capture frame-by-frame
    frame = cap.read()
    if p != 0:
        end_timer = time.time()
        t = end_timer - start_timer
    start_timer = time.time()
    
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
    
    mask_red = image_processing.image_process_red(frame,lower_red1,lower_red2,upper_red1,upper_red2)
    #res = cv2.bitwise_and(frame,frame, mask_red= mask_red)
    
    cnts_red = cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_red = cnts_red[0] if imutils.is_cv2() else cnts_red[1]
    sd = ShapeDetector()

    # loop over the contours
    for c in cnts_red:
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
                        i,sv,av,pcX,pcY,strss,ssmax,ssavg,vsmax,vsavg = velocity_math(i,t,sv,av,cX,cY,pcX,pcY,d,ang1,rat,rpph,rppv,hhfv,hvfv,riv_ang)
                        # multiply the contour (x, y)-coordinates by the resize ratio,
                        # then draw the contours and the name of the shape on the image
                        c = c.astype("int")
                        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                        cv2.putText(frame, strss, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    elif shape == "triangle":
                        h,tv,atv,ptcX,ptcY,strts,stmax,stavg,vtmax,vtavg = velocity_math(h,t,tv,atv,cX,cY,ptcX,ptcY,d,ang1,rat,rpph,rppv,hhfv,hvfv,riv_ang)
                        # multiply the contour (x, y)-coordinates by the resize ratio,
                        # then draw the contours and the name of the shape on the image
                        c = c.astype("int")
                        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                        cv2.putText(frame, strts, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    smax = max(ssmax,stmax)
    savg = (ssavg + stavg) / 2
    vmax = max(vsmax,vtmax)
    vavg = (vsavg + vtavg) / 2
    strsmax = "Maximum Speed = %f m/s" % smax
    strsavg = "Average Speed = %f m/s" % savg
    strvmax = "Maximum Speed = %f m/s" % vmax
    strvavg = "Average Speed = %f m/s" % vavg
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
    p = 1
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.stop()
cv2.destroyAllWindows()

np.savetxt("/home/pi/PartD_Project/Variables/red_sqr_vel.csv", np.array(av), delimiter=",")
np.savetxt("/home/pi/PartD_Project/Variables/red_sqr_speed.csv", np.array(sv), delimiter=",")
#np.savetxt("/home/pi/PartD_Project/Variables/avg_vel.csv", np.array(avg_vel), delimiter=",")