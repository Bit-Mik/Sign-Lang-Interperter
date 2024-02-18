import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
dect = HandDetector(maxHands=1)
#adding the path where the model and its label is saved
classifier = Classifier("E:\Programs\Python\HandSign Detector\Model\keras_model.h5","E:\Programs\Python\HandSign Detector\Model\labels.txt")

offset = 20
imgSize = (300)
labels=["A","B","C","D","E","F","G","H","I"]


while True:
    status, img = cap.read()
    hands, img = dect.findHands(img)

    #hands give the dict of info of the hand landmarks

    if hands:
        hand = hands[0]                                                #here we are making a new list "hand" and storing the data needed
        x, y, w, h = hand['bbox']                                      #assigning the x value,y value,w = width,h = height of our hand landmarks

        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255          #Creating a new image with the help of numpy module and using it to make a matrix
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]        #creating a new image window that will only show our hand

        imgCropShape = imgCrop.shape


        aspectRatio = h/w

        #making overlay Height is more then width
        #fixing height
        if aspectRatio > 1:                                                     #Height of pic is more than its width
            k = imgSize/h                                                       #creating a constant "k" for easy calculation
            wCal = math.ceil(k*w)                                               #wCal is the width of img we that is calculated
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))                      #resizing the image
            imgResizeShape = imgResize.shape
            wGap= math.ceil((imgSize-wCal)/2)                                   #for Centering the croped img in white BG 
            imgWhite[:, wGap:(wCal+wGap)] = imgResize                           #Centering

            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction,index)

        #Widht is more than height 
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:(hCal+hGap), :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction, index)

        cv2.rectangle(img, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0),cv2.FILLED) #creating the solid box where it will show all prediction
        cv2.putText(img, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

    cv2.imshow("Image",img)
    cv2.waitKey(1)


