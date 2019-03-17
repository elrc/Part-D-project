import cv2

vs = cv2.VideoCapture('example.avi')

while(vs.isOpened()):
    ret, frame = vs.read()
    
    if ret == True:
        
        cv2.imshow('frame',frame)
        
vs.release()
cv2.destroyAllWindows()