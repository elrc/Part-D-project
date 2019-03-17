from videostream import VideoStream
import imutils
import numpy as np
import cv2

def nothing(x):
    pass

def lower_red(cap):
    
    cv2.namedWindow("Trackbars")

    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 100, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 100, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 10, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
    
    while True:
        frame = cap.read()
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        
        value = cv2.split(hsv)[2]
        average1,_,_,_ = cv2.mean(value)
        
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")
        
        lower_red1 = np.array([l_h, l_s, l_v])
        lower_red2 = np.array([u_h, u_s, u_v])
        
        mask = cv2.inRange(hsv, lower_red1, lower_red2)
        
        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    return (lower_red1,lower_red2,average1)

def upper_red(cap):
    
    cv2.namedWindow("Trackbars")
    
    cv2.createTrackbar("L - H", "Trackbars", 160, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 100, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 100, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
    
    while True:
        frame = cap.read()
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        
        value = cv2.split(hsv)[2]
        average2,_,_,_ = cv2.mean(value)
        
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")
        
        upper_red1 = np.array([l_h, l_s, l_v])
        upper_red2 = np.array([u_h, u_s, u_v])
        
        mask = cv2.inRange(hsv, upper_red1, upper_red2)
        
        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()
    
    return (upper_red1,upper_red2,average2)

def blue(cap):
    
    cv2.namedWindow("Trackbars")

    cv2.createTrackbar("L - H", "Trackbars", 110, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 100, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 100, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 130, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
    
    while True:
        frame = cap.read()
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        
        value = cv2.split(hsv)[2]
        average3,_,_,_ = cv2.mean(value)
        
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")
        
        lower_blue = np.array([l_h, l_s, l_v])
        upper_blue = np.array([u_h, u_s, u_v])
        
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    return (lower_blue,upper_blue,average3)