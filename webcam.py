import cv2
cam = cv2.VideoCapture(0)
while True:
    retV, frame = cam.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    cv2.imshow('Maskcam',frame)
    cv2.imshow('Maskcam - grey', abu)
    k = cv2.waitKey(0) & 0xFF
    if k == 27 or k == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()