import cv2
import argparse
import numpy as np
from dataset import load_data
import matplotlib.pyplot as plt

# Utilisation de Matplotlib pour afficher les images
def show_image(image, title="Image"):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Analyse des arguments de la ligne de commande
parser = argparse.ArgumentParser()
parser.add_argument('--classifier', '-c', choices=['lbp', 'eigen', 'fisher'], default='lbp')
args = parser.parse_args()

# Chargement du modèle et des données de reconnaissance des visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Charger les données de formation et de test
(X_train, y_train), (X_test, y_test) = load_data(face_cascade)

# Création d'un système de reconnaissance des visages
if args.classifier == 'lbp':
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
elif args.classifier == 'eigen':
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
else:
    face_recognizer = cv2.face.FisherFaceRecognizer_create()

# Entraînement du système de reconnaissance sur les données d'entraînement
face_recognizer.train(X_train, np.array(y_train))

# Montrer quelques images d'entraînement pour vérification
for i in range(5):
    if i < len(X_train):
        show_image(X_train[i], f'Train Face {i}')

# Évaluer la précision du modèle sur des données de test
correct_predictions = 0
for i in range(len(X_test)):
    predicted_label, confidence = face_recognizer.predict(X_test[i])
    actual_label = y_test[i]
    
    if predicted_label == actual_label:
        correct_predictions += 1

# Calculer et imprimer la précision
accuracy = (correct_predictions / len(X_test)) * 100
print(f"Recognition accuracy: {accuracy:.2f}%")
