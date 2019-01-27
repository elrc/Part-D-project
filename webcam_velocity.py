from shapedetector import ShapeDetector
import imutils
import numpy as np
import cv2
import math

cap = cv2.VideoCapture('example.mp4')

pi = math.pi
tan = math.tan
cos = math.cos
sin = math.sin
sqrt = math.sqrt
atan = math.atan
hp = 1920
vp = 1080
fr = 30
t = 1/fr
d = 0.55

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

strvmax = "N/A"
strvavg = "N/A"

lower_red1 = np.array([0, 70, 50])
lower_red2 = np.array([10, 255, 255])
upper_red1 = np.array([170,70,50])
upper_red2 = np.array([180,255,255])

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = None
(h, w) = (None, None)

print("[INFO] starting analysis...")

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if writer is None:
        # store the image dimensions, initialzie the video writer,
        # and construct the eros array
        (h, w) = (1080, 1920)
        writer = cv2.VideoWriter("analysed.mp4", 0x00000021, 30,
            (w, h), True)

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
            if M['m00'] > 1000:
                cX = int((M['m10'] / M['m00']))
                cY = int((M['m01'] / M['m00']))
                if cX > 240 and cX < 1680:
                    if cY > 135 and cY < 945:
                        shape = sd.detect(c)
                        if shape == "square":
                            if i == 0:
                                pcX = cX
                                pcY = cY
                                strv = "0"
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
                                v = ds / t
                                strv = "%f m/s" % v
                                sv.append(v)
                                vmax = max(sv)
                                vavg = sum(sv) / len(sv)
                                strvmax = "Max Velocity = %f m/s" % vmax
                                strvavg = "Average Velocity = %f m/s" % vavg
                                pcX = cX
                                pcY = cY
                            # multiply the contour (x, y)-coordinates by the resize ratio,
                            # then draw the contours and the name of the shape on the image
                            c = c.astype("int")
                            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                            cv2.putText(frame, strv, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, strvmax, (1, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 0, 0), 2)
    cv2.putText(frame, strvavg, (1, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 0, 0), 2)
    
    writer.write(frame)
    
    # show the frame
    #cv2.imshow('frame',frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("[INFO] analysis finished")
cap.release()
cv2.destroyAllWindows()