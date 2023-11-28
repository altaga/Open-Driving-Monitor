import cv2
import numpy as np
import time
import pyautogui
import pygame
import json

widthScreen,heightScreen = pyautogui.size()
widthScreen,heightScreen=int(str(widthScreen)),int(str(heightScreen))-100

inputShape = (24,24)
eyesDetected = []
eyesState = []
fpsTimer = 0
start = 0
end = 0
loop = 0
flag = True

class_names = [ 'Close', 'Open' ]

jsonFile = {
    "state": True,
}

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
model = cv2.dnn.readNetFromTensorflow('blink-v1.pb')

# Camera
vid = cv2.VideoCapture(0) 
pygame.init()
pygame.mixer.music.load('alarm.mp3')

while(True):
    fpsTimer = time.time()
    ret, frame = vid.read() 
    # Preprocessing
    height, width, channels = frame.shape
    show = cv2.resize(frame, (widthScreen,heightScreen),  interpolation=cv2.INTER_AREA)
    faces = face_cascade.detectMultiScale(
        show,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = show[y:y + h, x:x + w]
        cv2.rectangle(show,(x,y),(x+w,y+h),(255,255,0),2)
        sensi=20
        eyes = eye_cascade.detectMultiScale(show,1.3, sensi)

        i=0
        for (x,y,w,h) in eyes:
            eye = show[y:y + h, x:x + w]
            eyeGray = cv2.cvtColor(eye, cv2.COLOR_RGB2GRAY)
            eyeGray3 = cv2.cvtColor(eyeGray, cv2.COLOR_GRAY2RGB)
            blob = cv2.dnn.blobFromImage(eyeGray3, 1/255.0, inputShape, swapRB=True, crop=False)
            cv2.rectangle(show,(x,y),(x+w,y+h),(255,255,255),2)
            eyesDetected.append(blob)

        if len(eyesDetected) > 0:
            for eye in eyesDetected:
                model.setInput(eye)
                outputs = model.forward(model.getUnconnectedOutLayersNames())
                final_outputs = outputs[0][0]
                label_id = np.argmax(final_outputs)
                eyesState.append(class_names[label_id])

            counter = 0
            for (x,y,w,h) in eyes:
                text = eyesState[counter]
                counter+=1
                cv2.putText(show, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            if 'Open' not in eyesState: 
                end = time.time()
            else:
                end = time.time()
                start = end

        else:
            end = time.time()

    else:
        end = time.time()

    eyesDetected = []
    eyesState = []
    fps = round(1/(time.time() - fpsTimer),2)
    cv2.putText(show, "FPS: "+str(fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)
    cv2.putText(show, "FPS: "+str(fps), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
    cv2.imshow('frame', show) 
    if((end - start) > 3): 
        print("Danger!")
        if flag :
            pygame.mixer.music.play(-1)
            flag = False
    else:
        print("Ok")
        flag = True
        pygame.mixer.music.stop()
    if(time.time()-loop > 5):
        loop = time.time()
        jsonFile["state"] = flag
        with open("drowsiness.json", "w") as outfile:
            json.dump(jsonFile, outfile)
        print("Saving")
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 