import cv2
import uuid

#images capture
cap = cv2.VideoCapture(-1)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
while True:
    ret, frame = cap.read()
    imgname = '/home/mrjaz/Pictures/Webcam/{}.jpg'.format(str(uuid.uuid1()))
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)

    if cv2.waitkey(1) & 0xFF == ord('q'):
        break
cap.release()
cap.destroyAllwindows()