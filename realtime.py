import cv2
import numpy as np
from dataset import load_data

# Chargement de l'outil de reconnaissance des visages et de l'ensemble de données
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
(X_train, y_train), _ = load_data(face_cascade)

# Créer un modèle de reconnaissance faciale
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(X_train, np.array(y_train))

# Initialisation de la webcam
cap = cv2.VideoCapture(0)
face_id_counter = 0
recognized_faces = {}

while True:
    # Capture image par image
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face_roi)
        
        if label not in recognized_faces:
            face_id_counter += 1
            recognized_faces[label] = face_id_counter
        
        # Afficher les résultats sur le cadre
        face_number = recognized_faces[label]
        frame = cv2.putText(frame, f"Face {face_number}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Afficher le cadre dans la fenêtre OpenCV
    cv2.imshow("Real-time Face Recognition", frame)

    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

# Lorsque tout est terminé, libérer la source
cap.release()
cv2.destroyAllWindows()
