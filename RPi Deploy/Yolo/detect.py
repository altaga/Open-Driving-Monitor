# YOLO object detection
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
import time

# Constants
minConfidence  = 0.5
classes = open('coco.names').read().strip().split('\n')
fpsTimer = 0

# Model
net = cv2.dnn.readNetFromDarknet('yolov3-tiny.cfg', 'yolov3-tiny.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Camera
vid = cv2.VideoCapture(0) 
  
while(True): 
    fpsTimer = time.time()
    # Utils
    confidences = []
    boxes = []
    classIDs = []
    # Capture the video frame 
    ret, frame = vid.read() 
    # Preprocessing
    height, width, channels = frame.shape
    show = cv2.resize(frame, (int(width/4),int(height/4)),  interpolation=cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(show, 1/255.0, (416, 416), swapRB=True, crop=False)
    # Network Processing
    net.setInput(blob)
    outputs = net.forward(ln)
    # Box Colors
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
    h, w = show.shape[:2]
    # Processing Result
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence >= minConfidence:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(show, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv2.putText(show, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            print(text)
  
    # Display the resulting frame 
    fps = round(1/(time.time() - fpsTimer),2)
    cv2.putText(show, "FPS: "+str(fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)
    cv2.putText(show, "FPS: "+str(fps), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
    cv2.imshow('frame', show) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 