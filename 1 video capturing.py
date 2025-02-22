import cv2
import numpy as np

cap=cv2.VideoCapture(r"D:\All Study Material\deep learning\opencv\object detection\highway.mp4")

while True:
    _,frame=cap.read()
    
    cv2.imshow('frame',frame)
    key=cv2.waitKey(1)
    if key==27:
        break
cap.release()
cap.destroyAllWindows()