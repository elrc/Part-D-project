# import the necessary packages
from videostream import VideoStream
import imutils
import numpy as np
import cv2
import time
import math

def nothing(x):
    pass

cap = VideoStream(usePiCamera=False, resolution=(960, 540)).start()
 
# allow the camera to warmup
time.sleep(0.1)

pi = math.pi
cos = math.cos
sin = math.sin

pt1x = 860
pt1y = 30

cv2.namedWindow("River Angle")

cv2.createTrackbar("Degrees", "River Angle", 90, 180, nothing)

while True:
    
    image = cap.read()
    
    ang = cv2.getTrackbarPos("Degrees", "River Angle")
    ang = ang - 90
    
    riv_ang = ang * (pi / 180)
    strrivang = "%d degrees" % round(ang)
    
    pt2x = round(pt1x+100*sin(riv_ang))
    pt2y = round(pt1y+100*cos(riv_ang))
    
    cv2.putText(image, strrivang, (pt1x-15, pt1y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 0, 255), 2)
    cv2.arrowedLine(image, (pt1x, pt1y), (pt2x, pt2y), (0, 0, 255), 3)
    
    cv2.imshow("Frame", image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()

np.savetxt("/home/pi/PartD_Project/Variables/river_angle.csv", np.array([ang]))