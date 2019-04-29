"""
Code to track and measure velocity of object using video file
Written by: Robert Crocker
Email contact: r.crocker-14@student.lboro.ac.uk
Extracts of code taken from:
https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
https://www.pyimagesearch.com/2016/01/04/unifying-picamera-and-cv2-videocapture-into-a-single-class-with-opencv/
Date created: 15/12/18
Last updated: 18/04/19
"""

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
import csv

# load video file
cap = cv2.VideoCapture('/home/pi/PartD_Project/Videos/example.avi')

#load variables
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

# grab arguments from command window
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cross", type=float, default=2,
    help="Cross-section of river in m^2")
args = vars(ap.parse_args())

cs = args["cross"]

# initialise the FourCC, video writer, dimensions of the frame, and zeros array
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = None
(hh, w) = (None, None)

# initialise maths
pi = math.pi
tan = math.tan
cos = math.cos
sin = math.sin
sqrt = math.sqrt
atan = math.atan
hp = 1024
vp = 576
fr = round(1/(sum(t)/len(t)))

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

# Initialise variable and arrays
i,h,p,pcX,pcY,ptcX,ptcY,smax,savg,vmax,vavg,ssmax,ssavg,vsmax,vsavg,stmax,stavg,vtmax,vtavg = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
e,ppcX,ppcY,spmax,spavg,vpmax,vpavg = 0,0,0,0,0,0,0
bi,bh,bpscX,bpscY,bptcX,bptcY,bssmax,bssavg,bvsmax,bvsavg,bstmax,bstavg,bvtmax,bvtavg = 0,0,0,0,0,0,0,0,0,0,0,0,0,0
be,bppcX,bppcY,bspmax,bspavg,bvpmax,bvpavg = 0,0,0,0,0,0,0
rsfrmax,rtfrmax,rpfrmax,bsfrmax,btfrmax,bpfrmax,rsfravg,rtfravg,rpfravg,bsfravg,btfravg,bpfravg,frmax,fravg = 0,0,0,0,0,0,0,0,0,0,0,0,0,0
strss,ssmax,ssavg,vsmax,vsavg,rsfrmax,rsfravg = "0",0,0,0,0,0,0
strts,stmax,stavg,vtmax,vtavg,rtfrmax,rtfravg = "0",0,0,0,0,0,0
strps,spmax,spavg,vpmax,vpavg,rpfrmax,rpfravg = "0",0,0,0,0,0,0
bstrss,bssmax,bssavg,bvsmax,bvsavg,bsfrmax,bsfravg = "0",0,0,0,0,0,0
bstrts,bstmax,bstavg,bvtmax,bvtavg,btfrmax,btfravg = "0",0,0,0,0,0,0
bstrps,bspmax,bspavg,bvpmax,bvpavg,bpfrmax,bpfravg = "0",0,0,0,0,0,0
rssl,rsvl,rsfrl,rsval,rtsl,rtvl,rtfrl,rtval,rpsl,rpvl,rpfrl,rpval = 0,0,0,0,0,0,0,0,0,0,0,0
bssl,bsvl,bsfrl,bsval,btsl,btvl,btfrl,btval,bpsl,bpvl,bpfrl,bpval = 0,0,0,0,0,0,0,0,0,0,0,0
trs,trt,trp,tbs,tbt,tbp,ctt = 0,0,0,0,0,0,0
trsa,trta,trpa,tbsa,tbta,tbpa,ct,at = [],[],[],[],[],[],[],[0]
sv,av,tv,atv,pv,apv = [],[],[],[],[],[]
bsv,bav,btv,batv,bpv,bapv = [],[],[],[],[],[]
rsva,rtva,rpva,bsva,btva,bpva = [],[],[],[],[],[]
rsf,rtf,rpf,bsf,btf,bpf = [],[],[],[],[],[]
svt,avt,rsft,rsvat,tvt,atvt,rtft,rtvat,pvt,apvt,rpft,rpvat = [],[],[],[],[],[],[],[],[],[],[],[]
bsvt,bavt,bsft,bsvat,btvt,batvt,btft,btvat,bpvt,bapvt,bpft,bpvat = [],[],[],[],[],[],[],[],[],[],[],[]

strsmax = "N/A"
strsavg = "N/A"
strvmax = "N/A"
strvavg = "N/A"

print("[INFO] framerate", fr, "fps")
print("[INFO] starting analysis...")

cv2.namedWindow("Window")

# loop whilst their are frames in the video file
while(cap.isOpened()):
    # grab frame from video file
    ret, frame = cap.read()
    
    if ret == True:
        # check if the writer is None
        if writer is None:
            # store the image dimensions, initialzie the video writer, and construct the eros array
            (hh, w) = (vp, hp)
            writer = cv2.VideoWriter("/home/pi/PartD_Project/Videos/analysed.mp4", 0x00000021, fr,
                (w, hh), True)
        
        d = distance[p]
        strd = "Height = %fm" % round(d, 2)
        
        # calculate time between last detected frame
        if i != 0: trs += t[p-1]
        if h != 0: trt += t[p-1]
        if e != 0: trp += t[p-1]
        if bi != 0: tbs += t[p-1]
        if bh != 0: tbt += t[p-1]
        if be != 0: tbp += t[p-1]
        if p != 0: ctt += t[p-1]
        ct.append(ctt)
        if p != 0: at.append(t[p-1])
        
        # find brightness of frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        value = cv2.split(image)[2]
        average,_,_,_ = cv2.mean(value)
        
        # adjust HSV boundaries depending on brightness comparison
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
            # compute the center of the contour, if in are range then detect the name of the shape using only the contour
            M = cv2.moments(c)
            if M['m00']>100:
                cX = int((M['m10'] / M['m00']))
                cY = int((M['m01'] / M['m00']))
                if cX > 128 and cX < 896:
                    if cY > 72 and cY < 504:
                        # calculates the shape of the contour
                        shape = sd.detect(c)
                        if shape == "square" and p != 0:
                            # variables are passed into the maths function to calculate speed, velocity and flow rate
                            i,sv,av,pcX,pcY,strss,ssmax,ssavg,vsmax,vsavg,rsf,rsfrmax,rsfravg,rsva = velocity_math(i,trs,sv,av,cX,cY,pcX,pcY,d,ang1,rat,rpph,rppv,hhfv,hvfv,riv_ang,cs,rsf,rsva,strss,
                                                                                                                   ssmax,ssavg,vsmax,vsavg,rsfrmax,rsfravg)
                            # time between detected frames is saved then reset
                            if trs != 0: trsa.append(trs)
                            trs = 0
                            # the found contour is drawn on the frame
                            c = c.astype("int")
                            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                            cv2.putText(frame, strss, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        elif shape == "triangle" and p != 0:
                            h,tv,atv,ptcX,ptcY,strts,stmax,stavg,vtmax,vtavg,rtf,rtfrmax,rtfravg,rtva = velocity_math(h,trt,tv,atv,cX,cY,ptcX,ptcY,d,ang1,rat,rpph,rppv,hhfv,hvfv,riv_ang,cs,rtf,rtva,
                                                                                                                      strts,stmax,stavg,vtmax,vtavg,rtfrmax,rtfravg)
                            if trt != 0: trta.append(trt)
                            trt = 0
                            c = c.astype("int")
                            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                            cv2.putText(frame, strts, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        elif shape == "pentagon" and p != 0:
                            e,pv,apv,ppcX,ppcY,strps,spmax,spavg,vpmax,vpavg,rpf,rpfrmax,rpfravg,rpva = velocity_math(e,trp,pv,apv,cX,cY,ppcX,ppcY,d,ang1,rat,rpph,rppv,hhfv,hvfv,riv_ang,cs,rpf,rpva,
                                                                                                                      strps,spmax,spavg,vpmax,vpavg,rpfrmax,rpfravg)
                            if trp != 0: trpa.append(trp)
                            trp = 0
                            c = c.astype("int")
                            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                            cv2.putText(frame, strps, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
        for c in cnts_blue:
            M = cv2.moments(c)
            if M['m00']>100:
                cX = int((M['m10'] / M['m00']))
                cY = int((M['m01'] / M['m00']))
                if cX > 128 and cX < 896:
                    if cY > 72 and cY < 504:
                        shape = sd.detect(c)
                        if shape == "square" and p != 0:
                            bi,bsv,bav,bpscX,bpscY,bstrss,bssmax,bssavg,bvsmax,bvsavg,bsf,bsfrmax,bsfravg,bsva = velocity_math(bi,tbs,bsv,bav,cX,cY,bpscX,bpscY,d,ang1,rat,rpph,rppv,hhfv,hvfv,riv_ang,
                                                                                                                               cs,bsf,bsva,bstrss,bssmax,bssavg,bvsmax,bvsavg,bsfrmax,bsfravg)
                            if tbs != 0: tbsa.append(tbs)
                            tbs = 0
                            c = c.astype("int")
                            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                            cv2.putText(frame, bstrss, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        elif shape == "triangle" and p != 0:
                            bh,btv,batv,bptcX,bptcY,bstrts,bstmax,bstavg,bvtmax,bvtavg,btf,btfrmax,btfravg,btva = velocity_math(bh,tbt,btv,batv,cX,cY,bptcX,bptcY,d,ang1,rat,rpph,rppv,hhfv,hvfv,riv_ang,
                                                                                                                                cs,btf,btva,bstrts,bstmax,bstavg,bvtmax,bvtavg,btfrmax,btfravg)
                            if tbt != 0: tbta.append(tbt)
                            tbt = 0
                            c = c.astype("int")
                            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                            cv2.putText(frame, bstrts, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        elif shape == "pentagon" and p != 0:
                            be,bpv,bapv,bppcX,bppcY,bstrps,bspmax,bspavg,bvpmax,bvpavg,bpf,bpfrmax,bpfravg,bpva = velocity_math(be,tbp,bpv,bapv,cX,cY,bppcX,bppcY,d,ang1,rat,rpph,rppv,hhfv,hvfv,riv_ang,
                                                                                                                                cs,bpf,bpva,bstrps,bspmax,bspavg,bvpmax,bvpavg,bpfrmax,bpfravg)
                            if tbp != 0: tbpa.append(tbp)
                            tbp = 0
                            c = c.astype("int")
                            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                            cv2.putText(frame, bstrps, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # calculate max and average for variables and display them on frame
        rss = 1 if ssavg != 0 else 0;rts = 1 if stavg != 0 else 0;rps = 1 if spavg != 0 else 0
        bss = 1 if bssavg != 0 else 0;bts = 1 if bstavg != 0 else 0;bps = 1 if bspavg != 0 else 0
        rsv = 1 if vsavg != 0 else 0;rtv = 1 if vtavg != 0 else 0;rpv = 1 if vpavg != 0 else 0
        bbsv = 1 if bvsavg != 0 else 0;bbtv = 1 if bvtavg != 0 else 0;bbpv = 1 if bvpavg != 0 else 0
        rsfrc = 1 if rsfravg != 0 else 0;rtfrc = 1 if rtfravg != 0 else 0;rpfrc = 1 if rpfravg != 0 else 0
        bsfrc = 1 if bsfravg != 0 else 0;btfrc = 1 if btfravg != 0 else 0;bpfrc = 1 if bpfravg != 0 else 0
        if rss != 0 or rts != 0 or rps != 0  or bss != 0 or bts != 0 or bps != 0:
            smax = round((max(ssmax,stmax,spmax,bssmax,bstmax,bspmax)),3)
            savg = round(((ssavg + stavg + spavg + bssavg + bstavg + bspavg) / (rss + rts + rps + bss + bts + bps)),3)
        if rsv != 0 or rtv != 0 or rpv != 0 or bbsv != 0 or bbtv != 0 or bbpv != 0:
            vmax = round((max(vsmax,vtmax,vpmax,bvsmax,bvtmax,bvpmax)),3)
            vavg = round(((vsavg + vtavg + vpavg + bvsavg + bvtavg + bvpavg) / (rsv + rtv + rpv + bbsv + bbtv + bbpv)),3)
        if rsfrc != 0 or rtfrc != 0 or rpfrc != 0 or bsfrc != 0 or btfrc != 0 or bpfrc != 0:
            frmax = round((max(rsfrmax,rtfrmax,rpfrmax,bsfrmax,btfrmax,bpfrmax)),3)
            fravg = round(((rsfravg + rtfravg + rpfravg + bsfravg + btfravg + bpfravg) / (rsfrc + rtfrc + rpfrc + bsfrc + btfrc + bpfrc)),3)
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
    
        # write frame to video file
        writer.write(frame)
        
        p += 1
        
        # write date to arrays
        if len(sv) > rssl: svt.append(sv[len(sv)-1]);rssl = len(sv)
        else: svt.append("")
        if len(av) > rsvl: avt.append(av[len(av)-1]);rsvl = len(av)
        else: avt.append("")
        if len(rsf) > rsfrl: rsft.append(rsf[len(rsf)-1]);rsfrl = len(rsf)
        else: rsft.append("")
        if len(rsva) > rsval: rsvat.append(rsva[len(rsva)-1]);rsval = len(rsva)
        else: rsvat.append("")
        if len(tv) > rtsl: tvt.append(tv[len(tv)-1]);rtsl = len(tv)
        else: tvt.append("")
        if len(atv) > rtvl: atvt.append(atv[len(atv)-1]);rtvl = len(atv)
        else: atvt.append("")
        if len(rtf) > rtfrl: rtft.append(rtf[len(rtf)-1]);rtfrl = len(rtf)
        else: rtft.append("")
        if len(rtva) > rtval: rtvat.append(rtva[len(rtva)-1]);rtval = len(rtva)
        else: rtvat.append("")
        if len(pv) > rpsl: pvt.append(pv[len(pv)-1]);rpsl = len(pv)
        else: pvt.append("")
        if len(apv) > rpvl: apvt.append(apv[len(apv)-1]);rpvl = len(apv)
        else: apvt.append("")
        if len(rpf) > rpfrl: rpft.append(rpf[len(rpf)-1]);rpfrl = len(rpf)
        else: rpft.append("")
        if len(rpva) > rpval: rpvat.append(rpva[len(rpva)-1]);rpval = len(rpva)
        else: rpvat.append("")
        if len(bsv) > bssl: bsvt.append(bsv[len(bsv)-1]);bssl = len(bsv)
        else: bsvt.append("")
        if len(bav) > bsvl: bavt.append(bav[len(bav)-1]);bsvl = len(bav)
        else: bavt.append("")
        if len(bsf) > bsfrl: bsft.append(bsf[len(bsf)-1]);bsfrl = len(bsf)
        else: bsft.append("")
        if len(bsva) > bsval: bsvat.append(bsva[len(bsva)-1]);bsval = len(bsva)
        else: bsvat.append("")
        if len(btv) > btsl: btvt.append(btv[len(btv)-1]);btsl = len(btv)
        else: btvt.append("")
        if len(batv) > btvl: batvt.append(batv[len(batv)-1]);btvl = len(batv)
        else: batvt.append("")
        if len(btf) > btfrl: btft.append(btf[len(btf)-1]);btfrl = len(btf)
        else: btft.append("")
        if len(btva) > btval: btvat.append(btva[len(btva)-1]);btval = len(btva)
        else: btvat.append("")
        if len(bpv) > bpsl: bpvt.append(bpv[len(bpv)-1]);bpsl = len(bpv)
        else: bpvt.append("")
        if len(bapv) > bpvl: bapvt.append(bapv[len(bapv)-1]);bpvl = len(bapv)
        else: bapvt.append("")
        if len(bpf) > bpfrl: bpft.append(bpf[len(bpf)-1]);bpfrl = len(bpf)
        else: bpft.append("")
        if len(bpva) > bpval: bpvat.append(bpva[len(bpva)-1]);bpval = len(bpva)
        else: bpvat.append("")
        
        if trs > 1: i = 0; trs = 0
        if trt > 1: h = 0; trt = 0
        if trp > 1: e = 0; trp = 0
        if tbs > 1: bi = 0; tbs = 0
        if tbt > 1: bh = 0; tbt = 0
        if tbp > 1: be = 0; tbp = 0
        
        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

print("[INFO] analysis finished")
# release camera
cap.release()
cv2.destroyAllWindows()

# svae arrays to CSV files
var_vel = np.array([smax,savg,vmax,vavg,frmax,fravg])
with open("/home/pi/PartD_Project/Variables/timedate.csv", "r") as readFile:
    reader = csv.reader(readFile)
    dates = list(reader)
data = zip(dates,at,ct,distance,rsvat,svt,avt,rsft,rtvat,tvt,atvt,rtft,rpvat,pvt,apvt,rpft,bsvat,bsvt,bavt,bsft,btvat,btvt,batvt,btft,bpvat,bpvt,bapvt,bpft)
with open("/home/pi/PartD_Project/Variables/data.csv", "w") as csvFile:
    file = csv.writer(csvFile)
    file.writerow(("Time Stamp","Time Between Frames","Cumulative Time","Height","Red Square Angle","Red Square Speed","Red Square Velocity","Red Square Flow Rate","Red Triangle Angle"
                   ,"Red Triangle Speed","Red Triangle Velocity","Red Triangle Flow Rate","Red Pentagon Angle","Red Pentagon Speed","Red Pentagon Velocity","Red Pentagon Flow Rate","Blue Square Angle"
                   ,"Blue Square Speed","Blue Square Velocity","Blue Square Flow Rate","Blue Triangle Angle","Blue Triangle Speed","Blue Triangle Velocity","Blue Triangle Flow Rate"
                   ,"Blue Pentagon Angle","Blue Pentagon Speed","Blue Pentagon Velocity","Blue Pentagon Flow Rate"))
    file.writerows(data)
readFile.close()
csvFile.close()
if av != []:
    AV = zip(trsa,av)
    with open("/home/pi/PartD_Project/Variables/red_sqr_vel.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(AV)
    csvFile.close()
if sv != []:
    SV = zip(trsa,sv)
    with open("/home/pi/PartD_Project/Variables/red_sqr_speed.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(SV)
    csvFile.close()
if rsf != []:
    RSF = zip(trsa,rsf)
    with open("/home/pi/PartD_Project/Variables/red_sqr_flow_rate.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(RSF)
    csvFile.close()
if rsva != []:
    RSVA = zip(trsa,rsva)
    with open("/home/pi/PartD_Project/Variables/red_sqr_vel_ang.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(RSVA)
    csvFile.close()
if atv != []:
    ATV = zip(trta,atv)
    with open("/home/pi/PartD_Project/Variables/red_tri_vel.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(ATV)
    csvFile.close()
if tv != []:
    TV = zip(trta,tv)
    with open("/home/pi/PartD_Project/Variables/red_tri_speed.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(TV)
    csvFile.close()
if rtf != []:
    RTF = zip(trta,rtf)
    with open("/home/pi/PartD_Project/Variables/red_tri_flow_rate.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(RTF)
    csvFile.close()
if rtva != []:
    RTVA = zip(trta,rtva)
    with open("/home/pi/PartD_Project/Variables/red_tri_vel_ang.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(RTVA)
    csvFile.close()
if apv != []:
    APV = zip(trpa,apv)
    with open("/home/pi/PartD_Project/Variables/red_pen_vel.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(APV)
    csvFile.close()
if pv != []:
    PV = zip(trpa,pv)
    with open("/home/pi/PartD_Project/Variables/red_pen_speed.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(PV)
    csvFile.close()
if rpf != []:
    RPF = zip(trpa,rpf)
    with open("/home/pi/PartD_Project/Variables/red_pen_flow_rate.csv", "w") as csvFile:
        file = csv.writer(csvFile)
        file.writerows(RPF)
    csvFile.close()
if rpva != []:
    RPVA = zip(trpa,rpva)
    with open("/home/pi/PartD_Project/Variables/red_pen_vel_ang.csv", "w") as csvFile:
        file = csv.writer(csvFile)
        file.writerows(RPVA)
    csvFile.close()
if bav != []:
    BAV = zip(tbsa,bav)
    with open("/home/pi/PartD_Project/Variables/blue_sqr_vel.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(BAV)
    csvFile.close()
if bsv != []:
    BSV = zip(tbsa,bsv)
    with open("/home/pi/PartD_Project/Variables/blue_sqr_speed.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(BSV)
    csvFile.close()
if bsf != []:
    BSF = zip(tbsa,bsf)
    with open("/home/pi/PartD_Project/Variables/blue_sqr_flow_rate.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(BSF)
    csvFile.close()
if bsva != []:
    BSVA = zip(tbsa,bsva)
    with open("/home/pi/PartD_Project/Variables/blue_sqr_vel_ang.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(BSVA)
    csvFile.close()
if batv != []:
    BATV = zip(tbta,batv)
    with open("/home/pi/PartD_Project/Variables/blue_tri_vel.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(BATV)
    csvFile.close()
if btv != []:
    BTV = zip(tbta,btv)
    with open("/home/pi/PartD_Project/Variables/blue_tri_speed.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(BTV)
    csvFile.close()
if btf != []:
    BTF = zip(tbta,btf)
    with open("/home/pi/PartD_Project/Variables/blue_tri_flow_rate.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(BTF)
    csvFile.close()
if btva != []:
    BTVA = zip(tbta,btva)
    with open("/home/pi/PartD_Project/Variables/blue_tri_vel_ang.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(BTVA)
    csvFile.close()
if bapv != []:
    BAPV = zip(tbpa,bapv)
    with open("/home/pi/PartD_Project/Variables/blue_pen_vel.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(BAPV)
    csvFile.close()
if bpv != []:
    BPV = zip(tbpa,bpv)
    with open("/home/pi/PartD_Project/Variables/blue_pen_speed.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(BPV)
    csvFile.close()
if bpf != []:
    BPF = zip(tbpa,bpf)
    with open("/home/pi/PartD_Project/Variables/blue_pen_flow_rate.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(BPF)
    csvFile.close()
if bpva != []:
    BPVA = zip(tbpa,bpva)
    with open("/home/pi/PartD_Project/Variables/blue_pen_vel_ang.csv", "w") as csvFile:
        file = csv.writer(csvFile);file.writerows(BPVA)
    csvFile.close()
label = ["Speed Max (m/s)","Speed Average (m/s)","Velocity Max (m/s)","Velocity Average (m/s)","Flow Rate Max (m^3/s)","Flow Rate Average (m^3/s)"]
veldata = zip(label,var_vel)
with open("/home/pi/PartD_Project/Variables/velocities.csv", "w") as csvFile:
    file = csv.writer(csvFile);file.writerows(veldata)
csvFile.close()

print("[INFO] program successfully executed")