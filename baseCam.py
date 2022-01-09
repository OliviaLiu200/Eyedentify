import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("trainner.yml")

# load label ids
lables = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()  # capture frame by frame
    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray frame

    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.5, minNeighbors=5)  # save images of face

    for (x, y, w, h) in faces:
        print(x, y, w, h)  # (y-cord to y-cord+height, x-cord to x-cord+width)
        roi_gray = gray_frame[y:y+h, x:x+w]

        # recognizing part
        id_, conf = face_recognizer.predict(roi_gray)
        if conf >= 55:
            print(id_)
            # put label on top of box
            id_font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(frame, labels[id_], (x, y-5), id_font, 1,
                        (255, 255, 255), 1, cv2.LINE_AA)
        else:
            print(id_)
            id_font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(frame, "unknown", (x, y-5), id_font, 1,
                        (255, 255, 255), 1, cv2.LINE_AA)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 0),
                      3)  # draw box around face

    # takes image of face
    #roi_actual = frame[y:y+h, x:x+w]
    #cv2.imwrite("me31.png", roi_actual)

    # display frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# release capture when complete
webcam.release()
cv2.destroyAllWindows()
