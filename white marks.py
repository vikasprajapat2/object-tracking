import cv2

cap = cv2.VideoCapture(r"D:\All Study Material\deep learning\opencv\object detection\highway.mp4")

object_detection = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)

while True:
    _,frame=cap.read()
    
    mask=object_detection.apply(frame)
    
    cv2.imshow('frame',frame)
    cv2.imshow('mask', mask)
    key=cv2.waitKey(30)
    if key==27:
        break
cap.release()
cap.destroyAllWindows()