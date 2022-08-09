import cv2
cam = cv2.VideoCapture(0)
cam.set(3, 648)#ubah lebar cam
cam.set(4, 488)#ubah tinggi cam
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    retV, frame = cam.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    faces = faceDetector.detectMultiScale(abu, 1.3, 5)# scale frame face
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('Maskcam',frame)
    #cv2.imshow('Maskcam - grey', abu)
    k = cv2.waitKey(0) & 0xFF
    if k == 27 or k == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()