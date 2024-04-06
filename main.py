#import the necessary libraries
import cv2
import numpy, math, time
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
import cvzone

# from keras.models import load_model

# classifier = load_model("Model/keras_model.h5", compile=False)
# class_names = open("Model/labels.txt", "r").readlines()


offset = 20
imgSize = 300

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
detector = HandDetector(maxHands=1)
face_detector = FaceDetector()



labels = ["Gesture"]

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break
    
    hands, frame = detector.findHands(frame)
    # frame, bboxs = FaceDetector.findFaces(frame, draw=True)
    # # lmList = detector.findPosition(frame, draw=False)

    # # if len(lmList) != 0:
    # #     x1, y1 = lmList[4][1], lmList[4][2]
    # #     x2, y2 = lmList[8][1], lmList[8][2]
    # #     cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    # #     cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
    # #     cv2.circle(frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
    # #     cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
    # #     cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
    # #     length = math.hypot(x2 - x1, y2 - y1)

    # if bboxs:
    #     # Loop through each bounding box
    #     for bbox in bboxs:
    #         # bbox contains 'id', 'bbox', 'score', 'center'

    #         # ---- Get Data  ---- #
    #         center = bbox["center"]
    #         x, y, w, h = bbox['bbox']
    #         score = int(bbox['score'][0] * 100)

    #         # ---- Draw Data  ---- #
    #         cv2.circle(frame, center, 5, (255, 0, 255), cv2.FILLED)
    #         cvzone.putTextRect(frame, f'{score}%', (x, y - 10))
    #         cvzone.cornerRect(frame, (x, y, w, h))
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        if len(hand['lmList']) != 0:
            x1, y1 = hand['lmList'][4][0], hand['lmList'][4][1]
            x2, y2 = hand['lmList'][8][0], hand['lmList'][8][1]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

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

            # imgResizeForModel = cv2.resize(imgWhite, (224, 224))
            # imgResizeForModel = numpy.expand_dims(imgResizeForModel, axis=0)
            # prediction = classifier.predict(imgResizeForModel)
            # index = numpy.argmax(prediction)
            # class_name = class_names[index]
            # confidence_score = prediction[0][index]

            # print("Class:", class_name[2:], end="")
            # print("Confidence Score:", str(numpy.round(confidence_score * 100))[:-2], "%")

            cv2.imshow("imgCrop", imgCrop)
            cv2.imshow("imgWhite", imgWhite)

            if cv2.waitKey(1) == ord('s'):
                print("Save")
                cv2.imwrite(f'Gestures/Gesture_{time.time()}.jpg', imgWhite)
        except:
            print("The hand must be shown entirely")

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if  key == ord('q'):
        break
    


cap.release()
cv2.destroyAllWindows()
