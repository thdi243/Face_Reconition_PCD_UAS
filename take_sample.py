import cv2
import os

# Make sure 'dataset' folder exists
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Make sure 'names' folder exists to save names
if not os.path.exists('names'):
    os.makedirs('names')

cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# For each person, enter one numeric face id
face_id = input('[ Take Sample ] enter user id (must be int) and press <return> ==>  ')
face_name = input('[ Take Sample ] enter user name and press <return> ==>  ')

# Save the name with corresponding ID
with open(f'names/{face_id}.txt', 'w') as f:
    f.write(face_name)

print("[ Take Sample ] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        print(f"[ Take Sample ] take sample count : {count + 1}")
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/sample." + str(face_id) + '.' +
                    str(count) + ".jpg", gray[y:y+h, x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 37:
        break
    elif count >= 40:  # Take 30 face sample and stop video
        break

# Do a bit of cleanup
print("[ Take Sample ] Exiting Program and cleanup stuff")

cam.release()
cv2.destroyAllWindows()
