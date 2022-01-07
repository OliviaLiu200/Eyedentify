import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()  # capture frame by frame
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray frame

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)
    # save images of face
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        # (y-cord to y-cord+height, x-cord to x-cord+width)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # recognizing part
        # confidence is not 0-100, not too sure what it is lol
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 50 and conf <= 90:
            print(id_)

        img_item = "its_me.png"
        cv2.imwrite(img_item, roi_gray)

        color = (100, 100, 0)  # color is in BGR, not RGB lol
        stroke = 3
        cv2.rectangle(frame, (x, y), (x+w, y+h), color,
                      stroke)  # draw box around face

    # display frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# release capture when complete
webcam.release()
cv2.destroyAllWindows()
