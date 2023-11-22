import cv2
import numpy as np
import time
import pyautogui
import pygame
import json

widthScreen,heightScreen = pyautogui.size()
widthScreen,heightScreen=int(str(widthScreen)),int(str(heightScreen))-100

inputShape = (48,48)
loop = 0
fpsTimer = 0

class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

jsonFile = {
    "emotion": "Neutral",
}

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = cv2.dnn.readNetFromTensorflow('emotions-v1.pb')

# Camera
vid = cv2.VideoCapture(0) 
pygame.init()
#pygame.mixer.music.load('alarm.mp3')

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
        blob = cv2.dnn.blobFromImage(face, 1/255.0, inputShape, swapRB=True, crop=False)
        model.setInput(blob)
        outputs = model.forward(model.getUnconnectedOutLayersNames())
        final_outputs = outputs[0][0]
        label_id = np.argmax(final_outputs)
        jsonFile["emotion"] = class_names[label_id]
    
    fps = round(1/(time.time() - fpsTimer),2)
    if(time.time()-loop > 5):
        loop = time.time()
        with open("emotion.json", "w") as outfile:
            json.dump(jsonFile, outfile)
        print("Saving")
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 