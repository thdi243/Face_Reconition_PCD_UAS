import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

# Create the LBPH face recognizer with custom parameters
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        # Apply Histogram Equalization to improve contrast
        img_numpy = cv2.equalizeHist(img_numpy)

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids

print("[ Train Image ] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
if not os.path.exists('trainer'):
    os.makedirs('trainer')

recognizer.write('trainer/trainer.yml')

# Print the number of faces trained and end program
print(f"[ Train Image ] {len(np.unique(ids))} faces trained. Exiting Program...")