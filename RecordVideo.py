# import the necessary packages
from __future__ import print_function
from videostream import VideoStream
import numpy as np
import imutils
import time
import cv2

# initialize the video stream and allow the camera
# sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=False).start()
time.sleep(2.0)
 
# initialize the FourCC, video writer, dimensions of the frame, and
# zeros array
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = None
(h, w) = (None, None)

# loop over frames from the video stream
while True:
    # grab the frame from the video stream and resize it to have a
    # maximum width of 300 pixels
    frame = vs.read()
 
    # check if the writer is None
    if writer is None:
        # store the image dimensions, initialzie the video writer,
        # and construct the eros array
        (h, w) = (576, 1024)
        writer = cv2.VideoWriter("example.avi", fourcc, 30,
            (w, h), True)
 
    output = frame
 
    # write the output frame to file
    writer.write(output)
    
    # show the frames
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
 
# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
writer.release()