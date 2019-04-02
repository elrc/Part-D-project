# import the necessary packages
from __future__ import print_function
from videostream import VideoStream
import RPi.GPIO as GPIO
import numpy as np
import warnings
import ultrasonic
import imutils
import time
import cv2

GPIO.setwarnings(False)

# initialize the video stream and allow the camera
# sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=False, resolution=(1024,576), framerate=20).start()
TRIG,ECHO = ultrasonic.ultrasonic_setup()

p = 0
dv,t = [],[]
 
# initialize the FourCC, video writer, dimensions of the frame, and
# zeros array
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = None
(h, w) = (None, None)

cv2.namedWindow("Window")

# loop over frames from the video stream
while True:
    # grab the frame from the video stream and resize it to have a
    # maximum width of 300 pixels
    frame = vs.read()
    if p != 0:
        end_timer = time.time()
        tt = end_timer - start_timer
        t.append(tt)
    start_timer = time.time()
    distance = ultrasonic.ultrasonic_read(TRIG,ECHO)
    dv.append(distance)
 
    # check if the writer is None
    if writer is None:
        # store the image dimensions, initialzie the video writer,
        # and construct the eros array
        (h, w) = (576, 1024)
        writer = cv2.VideoWriter("/home/pi/PartD_Project/Videos/example.avi", fourcc, 20,
            (w, h), True)
 
    # write the output frame to file
    writer.write(frame)
    
    p = 1
    
    # show the frames
    #cv2.imshow("Output", frame)
 
    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
writer.release()
ultrasonic.ultrasonic_cleanup()

np.savetxt("/home/pi/PartD_Project/Variables/distance_values.csv", np.array(dv), delimiter=",")
np.savetxt("/home/pi/PartD_Project/Variables/time_values.csv", np.array(t), delimiter=",")