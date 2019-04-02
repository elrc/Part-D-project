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

cap = cv2.VideoCapture('/home/pi/PartD_Project/Videos/example.avi')

distance = np.genfromtxt('/home/pi/PartD_Project/Variables/distance_values.csv', delimiter=',')
t = np.genfromtxt('/home/pi/PartD_Project/Variables/time_values.csv', delimiter=',')
river_angle = np.genfromtxt('/home/pi/PartD_Project/Variables/river_angle.csv')
river_angle.tolist()
lower_red1 = np.genfromtxt('/home/pi/PartD_Project/Variables/lower_red1.csv', delimiter=',')
lower_red2 = np.genfromtxt('/home/pi/PartD_Project/Variables/lower_red2.csv', delimiter=',')
upper_red1 = np.genfromtxt('/home/pi/PartD_Project/Variables/upper_red1.csv', delimiter=',')
upper_red2 = np.genfromtxt('/home/pi/PartD_Project/Variables/upper_red2.csv', delimiter=',')
lower_blue = np.genfromtxt('/home/pi/PartD_Project/Variables/lower_blue.csv', delimiter=',')
upper_blue = np.genfromtxt('/home/pi/PartD_Project/Variables/upper_blue.csv', delimiter=',')
averages = np.genfromtxt('/home/pi/PartD_Project/Variables/averages.csv', delimiter=',')
average1 = averages[0]
average2 = averages[1]
average3 = averages[2]

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cross", type=float, default=2,
    help="Cross-section of river in m^2")
args = vars(ap.parse_args())

cs = args["cross"]

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
fr =round(1/(sum(t)/len(t)))

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

i,h,p,pcX,pcY,ptcX,ptcY,smax,savg,vmax,vavg,ssmax,ssavg,vsmax,vsavg,stmax,stavg,vtmax,vtavg = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
bi,bh,bpscX,bpscY,bptcX,bptcY,bssmax,bssavg,bvsmax,bvsavg,bstmax,bstavg,bvtmax,bvtavg = 0,0,0,0,0,0,0,0,0,0,0,0,0,0
rsfrmax,rtfrmax,bsfrmax,btfrmax,rsfravg,rtfravg,bsfravg,btfravg,frmax,fravg = 0,0,0,0,0,0,0,0,0,0
trs,trt,tbs,tbt = 0,0,0,0
sv,av,tv,atv = [],[],[],[]
bsv,bav,btv,batv = [],[],[],[]
rsf,rtf,bsf,btf = [],[],[],[]

strsmax = "N/A"
strsavg = "N/A"
strvmax = "N/A"
strvavg = "N/A"

print("[INFO] framerate", fr)
print("[INFO] starting analysis...")

cv2.namedWindow("Window")

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:
    
        if writer is None:
            # store the image dimensions, initialzie the video writer,
            # and construct the eros array
            (h, w) = (576, 1024)
            writer = cv2.VideoWriter("/home/pi/PartD_Project/Videos/analysed.mp4", 0x00000021, fr,
                (w, h), True)
        
        d = distance[p]
        strd = "Height = %fm" % round(d, 4)
        
        if i != 0:
            trs += t[p-1]
        if h != 0:
            trt += t[p-1]
        if bi != 0:
            tbs += t[p-1]
        if bh != 0:
            tbt += t[p-1]
        
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
        if average != average3:
            diff3 = average - average3
            lower_blue[2] = lower_blue[2] + round(diff3)
            average3 = average
        
        cnts_red = image_processing.image_process_red(frame,lower_red1,lower_red2,upper_red1,upper_red2)
        cnts_blue = image_processing.image_process(frame,lower_blue,upper_blue)
        
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
                        if shape == "square" and p != 0:
                            i,sv,av,pcX,pcY,strss,ssmax,ssavg,vsmax,vsavg,rsf,rsfrmax,rsfravg = velocity_math(i,trs,sv,av,cX,cY,pcX,pcY,d,ang1,rat,rpph,rppv,hhfv,hvfv,riv_ang,cs,rsf)
                            trs = 0
                            # multiply the contour (x, y)-coordinates by the resize ratio,
                            # then draw the contours and the name of the shape on the image
                            c = c.astype("int")
                            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                            cv2.putText(frame, strss, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        elif shape == "triangle" and p != 0:
                            h,tv,atv,ptcX,ptcY,strts,stmax,stavg,vtmax,vtavg,rtf,rtfrmax,rstravg = velocity_math(h,trt,tv,atv,cX,cY,ptcX,ptcY,d,ang1,rat,rpph,rppv,hhfv,hvfv,riv_ang,cs,rtf)
                            trt = 0
                            # multiply the contour (x, y)-coordinates by the resize ratio,
                            # then draw the contours and the name of the shape on the image
                            c = c.astype("int")
                            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                            cv2.putText(frame, strts, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
        for c in cnts_blue:
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            M = cv2.moments(c)
            if M['m00']>100:
                cX = int((M['m10'] / M['m00']))
                cY = int((M['m01'] / M['m00']))
                if cX > 128 and cX < 896:
                    if cY > 72 and cY < 504:
                        shape = sd.detect(c)
                        if shape == "square" and p != 0:
                            bi,bsv,bav,bpscX,bpscY,bstrss,bssmax,bssavg,bvsmax,bvsavg,bsf,bsfrmax,bsfravg = velocity_math(bi,tbs,bsv,bav,cX,cY,bpscX,bpscY,d,ang1,rat,rpph,rppv,hhfv,hvfv,riv_ang,cs,bsf)
                            tbs = 0
                            # multiply the contour (x, y)-coordinates by the resize ratio,
                            # then draw the contours and the name of the shape on the image
                            c = c.astype("int")
                            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                            cv2.putText(frame, bstrss, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        elif shape == "triangle" and p != 0:
                            bh,btv,batv,bptcX,bptcY,bstrts,bstmax,bstavg,bvtmax,bvtavg,btf,btfrmax,btfravg = velocity_math(bh,tbt,btv,batv,cX,cY,bptcX,bptcY,d,ang1,rat,rpph,rppv,hhfv,hvfv,riv_ang,cs,btf)
                            tbt = 0
                            # multiply the contour (x, y)-coordinates by the resize ratio,
                            # then draw the contours and the name of the shape on the image
                            c = c.astype("int")
                            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                            cv2.putText(frame, bstrts, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
        rss = 1 if ssavg != 0 else 0
        rts = 1 if stavg != 0 else 0
        bss = 1 if bssavg != 0 else 0
        bts = 1 if bstavg != 0 else 0
        rsv = 1 if vsavg != 0 else 0
        rtv = 1 if vtavg != 0 else 0
        bbsv = 1 if bvsavg != 0 else 0
        bbtv = 1 if bvtavg != 0 else 0
        rsfrc = 1 if rsfravg != 0 else 0
        rtfrc = 1 if rtfravg != 0 else 0
        bsfrc = 1 if bsfravg != 0 else 0
        btfrc = 1 if btfravg != 0 else 0
        if rss != 0 or rts != 0 or bss != 0 or bts != 0:
            smax = max(ssmax,stmax,bssmax,bstmax)
            savg = (ssavg + stavg + bssavg + bstavg) / (rss + rts + bss + bts)
        if rsv != 0 or rtv != 0 or bbsv != 0 or bbtv != 0:
            vmax = max(vsmax,vtmax,bvsmax,bvtmax)
            vavg = (vsavg + vtavg + bvsavg + bvtavg) / (rsv + rtv + bbsv + bbtv)
        if rsfrc != 0 or rtfrc != 0 or bsfrc != 0 or btfrc != 0:
            frmax = max(rsfrmax,rtfrmax,bsfrmax,btfrmax)
            fravg = (rsfravg + rtfravg + bsfravg + btfravg) / (rsfrc + rtfrc + bsfrc + btfrc)
        strsmax = "Maximum Speed = %f m/s" % smax
        strsavg = "Average Speed = %f m/s" % savg
        strvmax = "Maximum Velocity = %f m/s" % vmax
        strvavg = "Average Velocity = %f m/s" % vavg
        strfrmax = "Maximum Flow Rate = %f m^3/s" % frmax
        strfravg = "Average Flow Rate = %f m^3/s" % fravg
        cv2.putText(frame, strsmax, (1, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 0, 0), 2)
        cv2.putText(frame, strsavg, (1, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 0, 0), 2)
        cv2.putText(frame, strvmax, (1, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 0, 0), 2)
        cv2.putText(frame, strvavg, (1, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 0, 0), 2)
        cv2.putText(frame, strfrmax, (1, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 0, 0), 2)
        cv2.putText(frame, strfravg, (1, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 0, 0), 2)
        cv2.putText(frame, strd, (1, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 0, 0), 2)
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

var_vel = np.array([smax,savg,vmax,vavg,frmax,fravg])

if av != []: np.savetxt("/home/pi/PartD_Project/Variables/red_sqr_vel.csv", np.array(av), delimiter=",")
if sv != []: np.savetxt("/home/pi/PartD_Project/Variables/red_sqr_speed.csv", np.array(sv), delimiter=",")
if rsf != []: np.savetxt("/home/pi/PartD_Project/Variables/red_sqr_flow_rate.csv", np.array(rsf), delimiter=",")
if atv != []: np.savetxt("/home/pi/PartD_Project/Variables/red_tri_vel.csv", np.array(atv), delimiter=",")
if tv != []: np.savetxt("/home/pi/PartD_Project/Variables/red_tri_speed.csv", np.array(tv), delimiter=",")
if rtf != []: np.savetxt("/home/pi/PartD_Project/Variables/red_tri_flow_rate.csv", np.array(rtf), delimiter=",")
if bav != []: np.savetxt("/home/pi/PartD_Project/Variables/blue_sqr_vel.csv", np.array(bav), delimiter=",")
if bsv != []: np.savetxt("/home/pi/PartD_Project/Variables/blue_sqr_speed.csv", np.array(bsv), delimiter=",")
if bsf != []: np.savetxt("/home/pi/PartD_Project/Variables/blue_sqr_flow_rate.csv", np.array(bsf), delimiter=",")
if batv != []: np.savetxt("/home/pi/PartD_Project/Variables/blue_tri_vel.csv", np.array(batv), delimiter=",")
if btv != []: np.savetxt("/home/pi/PartD_Project/Variables/blue_tri_speed.csv", np.array(btv), delimiter=",")
if btf != []: np.savetxt("/home/pi/PartD_Project/Variables/blue_tri_flow_rate.csv", np.array(btf), delimiter=",")
np.savetxt("/home/pi/PartD_Project/Variables/velocities.csv", var_vel, delimiter=",")