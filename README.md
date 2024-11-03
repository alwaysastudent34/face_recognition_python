# face_recognition_python
Real-Time Face Recognition
This project implements real-time face recognition using OpenCV and a pre-trained face recognition model. The program is capable of detecting and recognizing faces in video streams from a webcam. It provides options for face classification using different recognition algorithms.

Features
Real-time face detection and recognition using a webcam
Face classification with Local Binary Patterns (LBP), Eigenfaces, and Fisherfaces algorithms
Integration with the Yale Faces dataset for training and testing
Requirements
To run this project, you'll need the following dependencies:

Python 3.x
OpenCV
NumPy
Matplotlib
PIL
Install all required dependencies with:
pip install -r requirements.txt

Project Structure
app.py: Script for training the face recognition model on the Yale Faces dataset.
dataset.py: Handles loading and processing the face images from the dataset directory.
realtime.py: Main script for real-time face recognition using a webcam.
yalefaces/: Folder expected to contain the Yale Faces dataset images (not included due to licensing).
Usage
1. Prepare the Dataset
Download the Yale Faces dataset from the official source and place it in the yalefaces directory within the project folder. The folder structure should look like this:

yalefaces/
    subject01.gif
    subject01.wink.gif
    subject02.gif
    ...


Ensure that non-wink images are used for training, and wink images are reserved for testing.

2. Train the Model
To train the face recognition model with the dataset, run:
python app.py --classifier lbp
    ou can also select eigen or fisher for different face classification algorithms.

3. Run Real-Time Face Recognition
Once the model is trained, start the real-time face recognition by executing:
python realtime.py
Note on Dataset Licensing
Due to licensing constraints, the Yale Faces dataset is not included in this repository. You can download it from the official source and place it in the yalefaces directory.

Acknowledgments
OpenCV: For providing the tools for image processing and face recognition.
Yale University: For the Yale Faces dataset used in training and testing.
