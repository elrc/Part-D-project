from videostream import VideoStream
import colour_detect
import cv2
import numpy as np

def nothing(x):
    pass
 
cap = VideoStream(usePiCamera=False).start()

lower_red1,lower_red2,average1 = colour_detect.lower_red(cap)
upper_red1,upper_red2,average2 = colour_detect.upper_red(cap)
lower_blue,upper_blue,average3 = colour_detect.blue(cap)
averages = np.array([average1, average2, average3])
    
cap.stop()

np.savetxt("/home/pi/PartD_Project/Variables/lower_red1.csv", lower_red1, delimiter=",")
np.savetxt("/home/pi/PartD_Project/Variables/lower_red2.csv", lower_red2, delimiter=",")
np.savetxt("/home/pi/PartD_Project/Variables/upper_red1.csv", upper_red1, delimiter=",")
np.savetxt("/home/pi/PartD_Project/Variables/upper_red2.csv", upper_red2, delimiter=",")
np.savetxt("/home/pi/PartD_Project/Variables/lower_blue.csv", lower_blue, delimiter=",")
np.savetxt("/home/pi/PartD_Project/Variables/upper_blue.csv", upper_blue, delimiter=",")
np.savetxt("/home/pi/PartD_Project/Variables/averages.csv", averages, delimiter=",")