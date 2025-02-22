import cv2
from tracker import *  #costum file

tracker = EuclideanDistTracker() # tracker are object to check the position of object


cap = cv2.VideoCapture(r"D:\All Study Material\deep learning\opencv\object detection\highway.mp4")

#object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)  #used the backegraind , history =100 -- 100 frames histry mainten , varthreshold = 40 --- modition detection 


while True:
    ret, frame = cap.read()
    height, width , _ = frame.shape
    
    #Extract Region of interest
    roi = frame[340: 720, 500: 800]
    
    #object detection
    mask = object_detector.apply(roi)  #background delete and detect moving object
    _, mask = cv2.threshold(mask, 254 ,255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # cv2.findContours(): White (foreground) objects ke contours (boundaries) detect karega. ,cv2.RETR_TREE: Contour hierarchy maintain karta hai. ,cv2.CHAIN_APPROX_SIMPLE: Extra points ko remove karta hai (memory save karta hai).
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt) #calculate area and remove small elements
        if area > 100:
            x , y, w, h = cv2.boundingRect(cnt)
            detections.append([x,y,w,h])
    
    #object tracking
    boxes_ids = tracker.update(detections)
    for box_ids in boxes_ids:
        x, y,w,h ,id = box_ids
        #text on the object
        cv2.putText(roi, str(id), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 2, (255 , 0,0),2)
        cv2.rectangle(roi, (x,y), (x+w,y+h),(0,255,0),3)
        
    cv2.imshow('roi', roi)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    
    key=cv2.waitKey(40)
    if key == 27:
        break
cap.release()
cv2.distroyAllWindwows()
            