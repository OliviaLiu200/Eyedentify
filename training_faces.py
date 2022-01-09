import cv2
import os
import numpy as np
import pickle
from PIL import Image

face_recognizer = cv2.face.LBPHFaceRecognizer_create()  # declaring LBPH recognizer
face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "photos")

x_training = []
y_label = []
id_cur = 0
labels = {}

# os.walk for image finding
for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("PNG") or file.endswith("jpg") or file.endswith("JPG"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(
                " ", "-").lower()  # labels from directory
            # print(path)
            if label in labels:
                pass
            else:
                labels[label] = id_cur
                id_cur += 1
            id = labels[label]
            # print(labels)

            # train images to numpy arrays
            image = Image.open(path).convert("L").resize(
                (600, 600), Image.ANTIALIAS)  # grayscale and resized for training

            photos_array = np.array(image, "uint8")  # print(image_array)

            # find region of interests in training photos
            faces = face_cascade.detectMultiScale(
                photos_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = photos_array[y:y+h, x:x+h]
                x_training.append(roi)
                y_label.append(id)

# print(x_training)
# print(y_label)

# save label ids
with open("labels.pickle", 'wb') as f:
    pickle.dump(labels, f)

face_recognizer.train(x_training, np.array(y_label))
face_recognizer.save("trainner.yml")
