#import the necessary libraries
import cv2
import mediapipe as mp
import numpy, math
from cvzone.HandTrackingModule import HandDetector


offset = 20
imgSize = 300

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
detector = HandDetector(maxHands=1)


while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break
    
    hands, frame = detector.findHands(frame)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        try:

            imgCrop = frame[y - offset: y + h + offset, x - offset: x + w + offset]
            imgWhite = numpy.ones((imgSize, imgSize, 3), numpy.uint8) * 255

            imgCropShape = imgCrop.shape

            aspectratio = h / w

            if aspectratio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((300 - wCal) / 2)
                imgWhite[0: imgResizeShape[0], wGap: imgResizeShape[1] + wGap] = imgResize
            
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((300 - hCal) / 2)
                imgWhite[hGap: imgResizeShape[0] + hGap, 0: imgResizeShape[1]] = imgResize

            cv2.imshow("imgCrop", imgCrop)
            cv2.imshow("imgWhite", imgWhite)
        except:
            print("The hand must be shown entirely")
        

    cv2.imshow('frame', frame)
    if  cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()