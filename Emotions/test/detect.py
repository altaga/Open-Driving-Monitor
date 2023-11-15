import cv2
import numpy as np
import time
import os
import json

precisionByClass = {}
counter = 0

class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
for classes in class_names:
    precisionByClass[classes] = 0

testImages = sorted(os.listdir("testImages"))

model = cv2.dnn.readNetFromTensorflow('../../emotions-v1.pb')

# load the image from disk

for imagePath in testImages:
    startReal = time.process_time()
    image = cv2.imread("testImages/"+imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(gray3, (48, 48),  interpolation=cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(image=resized, scalefactor=1/256, size=(48, 48))
    model.setInput(blob)
    start = time.process_time()
    outputs = model.forward(model.getUnconnectedOutLayersNames())
    print("{:.4f} Processing Seconds".format(time.process_time() - start))
    final_outputs = outputs[0][0]
    label_id = np.argmax(final_outputs)
    probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
    final_prob = np.max(probs) * 100.
    out_name = class_names[label_id]
    print("{:.4f} Real Seconds".format(time.process_time() - startReal))
    print("Network Detection:{} : Real Label: {} : File {}".format(out_name, imagePath.replace(".png","")[:-1],imagePath))
    if imagePath.replace(".png","")[:-1]== out_name : counter +=1
    if imagePath.replace(".png","")[:-1]== out_name : precisionByClass[out_name] +=1

for classes in class_names:
    precisionByClass[classes] /= 4

print(json.dumps(precisionByClass))
print("Network Precision:{:.2f}%".format((counter/len(testImages))*100))