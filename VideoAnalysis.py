from shapedetector import ShapeDetector
import imutils
import numpy as np
import cv2

cap = cv2.VideoCapture('example.mp4')

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
            if M['m00']>100:
                cX = int((M['m10'] / M['m00']))
                cY = int((M['m01'] / M['m00']))
                shape = sd.detect(c)
 
                # multiply the contour (x, y)-coordinates by the resize ratio,
                # then draw the contours and the name of the shape on the image
                c = c.astype("float")
                c = c.astype("int")
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    writer.write(frame)
    
    # show the frame
    #cv2.imshow('frame',frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("[INFO] analysis finished")
cap.release()
cv2.destroyAllWindows()