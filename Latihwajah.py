import cv2, os, numpy as np
from PIL import Image

wajahDir = 'datawajah'
latihDir = 'latihwajah'

def getImageLabel(path):
    imagePath = [os.path.join(path,f) for f in os.listdir(path)]
    faceSample = []
    faceIDs = []
    for imagePath in imagePath:
        PILImg = Image.open(imagePath).convert('L')
        imgNum = np.array(PILImg,'uint8')
        faceID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = faceDetector.detectMultiScale(imgNum)
        for(x, y, w, h) in faces:
            faceSample.append(imgNum[y:y+h,x:x+w])
            faceIDs.append(faceID)
        return faceSample,faceIDs

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print("This machine is still learning, please wait")
faces,IDs = getImageLabel(wajahDir)
faceRecognizer.train(faces,np.array(IDs))

#simpan muka
faceRecognizer.write(latihDir+'/training.xml')
print('many {0} data wajah telah ditrainingkan ke mesin.',format(len(np.unique(IDs))))

