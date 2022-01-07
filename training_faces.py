import cv2
import os
import numpy as np
import pickle
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "photos")

recognizer = cv2.face.LBPHFaceRecognizer_create()  # declaring LBPH recognizer

x_train = []
y_labels = []
cur_id = 0
label_ids = {}

face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')

# os.walk for image finding
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("PNG") or file.endswith("JPG"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(
                " ", "-").lower()  # labels from directory
            # print(path)
            if label in label_ids:
                pass
            else:
                label_ids[label] = cur_id
                cur_id += 1

            id_ = label_ids[label]
            # print(label_ids)

            # train images to numpy arrays
            pil_image = Image.open(path).convert("L")  # grayscale
            image_array = np.array(pil_image, "uint8")
            # print(image_array)

            # find region of interests in training photos
            faces = face_cascade.detectMultiScale(
                image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+h]
                x_train.append(roi)
                y_labels.append(id_)

# print(y_labels)
# print(x_train)

# save label ids
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
