import cv2
import numpy as np
import time

class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

model = cv2.dnn.readNetFromTensorflow('emotions-v1.pb')

# load the image from disk
image = cv2.imread('smiling.png')
color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized = cv2.resize(color, (48, 48),  interpolation=cv2.INTER_AREA)
# create blob from image
start = time.process_time()
blob = cv2.dnn.blobFromImage(image=resized, scalefactor=1/256, size=(48, 48))
# set the input blob for the neural network
model.setInput(blob)
# forward pass image blog through the model
outputs = model.forward(model.getUnconnectedOutLayersNames())
# Get Values
final_outputs = outputs[0][0]
# get the class label
label_id = np.argmax(final_outputs)
# convert the output scores to softmax probabilities
probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
# get the final highest probability
final_prob = np.max(probs) * 100.
# map the max confidence to the class label names
out_name = class_names[label_id]
print("{:.4f} Seconds".format(time.process_time() - start))
print("Detection:{}".format(out_name))
