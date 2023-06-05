from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture("Fall-detection\\video.mp4")
cap.set(3, 1280)
cap.set(4, 720)
model = YOLO("YOLO-PretrainedModels\\yolov8n.pt")
classnames = ["person"]

fall_detected=False
fall_start_time=None

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            print(x1, y1, x2, y2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2-x1, y2-y1
            bbox = int(x1), int(y1), int(w), int(h)

            conf = math.ceil((box.conf[0]*100))/100
        
            fall_detected=False
            cls = box.cls[0]
            if cls == 0 and conf > 0.3 and int(w)>int(h):
                fall_detected=True
                cvzone.cornerRect(img, bbox, l=9)
                cvzone.putTextRect(img, f'Fall Detected {conf}', (max(
                    0, x1), max(35, y1)), scale=1, thickness=1, offset=3)
                
                
                if fall_detected and fall_start_time is None:
                    fall_start_time=time.time()
                if not fall_detected and fall_start_time is not None:
                    fall_start_time=None
                if fall_detected and fall_start_time is not None and fall_start_time>10:
                    #Write the code for sending the mail here!!

                    print("Mail Sended!!")
            if not fall_detected and fall_start_time is not None:
                fall_start_time = None


    cv2.imshow("Image", img)
    cv2.waitKey(0)
