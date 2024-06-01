# initialize project
import os

print("[ Initialize Project ] Check if folder exist, if not create it")
dataset = os.path.exists('dataset')
trainer = os.path.exists('trainer')

if not dataset:
    os.makedirs('dataset')

if not trainer:
    os.makedirs('trainer')

# download haarcascade pre-trained model
import urllib.request

print("[ Initialize Project ] Downloading haarcascade pre-trained model")
url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
filename = 'haarcascade_frontalface_default.xml'

if not os.path.exists(filename):
    urllib.request.urlretrieve(url, filename)