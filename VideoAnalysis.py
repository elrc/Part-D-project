from __future__ import print_function
from shapedetector import ShapeDetector
from maths import velocity_math
import image_processing
import imutils
import numpy as np
import cv2
import time
import math
import argparse

cap = cv2.VideoCapture('example.avi')

distance = np.genfromtxt('/home/pi/PartD_Project/Variables/distance_values.csv', delimiter=',')
river_angle = np.genfromtxt('/home/pi/PartD_Project/Variables/river_angle.csv')
river_angle.tolist()
lower_red1 = np.genfromtxt('/home/pi/PartD_Project/Variables/lower_red1.csv', delimiter=',')
lower_red2 = np.genfromtxt('/home/pi/PartD_Project/Variables/lower_red2.csv', delimiter=',')
upper_red1 = np.genfromtxt('/home/pi/PartD_Project/Variables/upper_red1.csv', delimiter=',')
upper_red2 = np.genfromtxt('/home/pi/PartD_Project/Variables/upper_red2.csv', delimiter=',')
averages = np.genfromtxt('/home/pi/PartD_Project/Variables/averages.csv', delimiter=',')
average1 = averages[0]
average2 = averages[1]

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = None
(h, w) = (None, None)

pi = math.pi
tan = math.tan
cos = math.cos
sin = math.sin
sqrt = math.sqrt
atan = math.atan
hp = 1024
vp = 576
fr = 30
t = 1/fr

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

i = 0
p = 0
pcX = 0
pcY = 0
sv = []
av = []

strsmax = "N/A"
strsavg = "N/A"
strvmax = "N/A"
strvavg = "N/A"

print("[INFO] starting analysis...")

cv2.namedWindow("Window")

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:
    
        if writer is None:
            # store the image dimensions, initialzie the video writer,
            # and construct the eros array
            (h, w) = (576, 1024)
            writer = cv2.VideoWriter("analysed.mp4", 0x00000021, 20,
                (w, h), True)
        
        d = distance[p]
        strd = "%fm" % round(d, 4)
        
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
        #res = cv2.bitwise_and(frame,frame, mask= mask)
    
        cnts_red = cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_red = cnts[0] if imutils.is_cv2() else cnts_red[1]
        sd = ShapeDetector()

        # loop over the contours
        for c in cnts_red:
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            M = cv2.moments(c)
            if M['m00']>100:
                cX = int((M['m10'] / M['m00']))
                cY = int((M['m01'] / M['m00']))
                if cX > 128 and cX < 896:
                    if cY > 72 and cY < 504:
                        shape = sd.detect(c)
                        if shape == "square":
                            i,sv,av,pcX,pcY,strs,strsmax,strsavg,strvmax,strvavg = velocity_math(i,t,sv,av,cX,cY,pcX,pcY,d,ang1,rat,rpph,rppv,hhfv,hvfv,riv_ang)
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
    
        writer.write(frame)
        
        p += 1
    
        # show the frame
        #cv2.imshow('frame',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    else:
        break

print("[INFO] analysis finished")
cap.release()
cv2.destroyAllWindows()