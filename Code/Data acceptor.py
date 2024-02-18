import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)
dect = HandDetector(maxHands=1)

offset = 20
imgSize = (300)

folder = "E:\Programs\Python\HandSign Detector\DATA\A"
counter = 0

while True:
    status, img = cap.read()
    hands, img = dect.findHands(img)

    #hands give the dict of info of the hand landmarks

    if hands:
        hand = hands[0]                                         #here we are making a new list "hand" and storing the data needed
        x, y, w, h = hand['bbox']                               #assigning the x value,y value,w = width,h = height of our hand landmarks

        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255   #Creating a new image with the help of numpy module and using it to make a matrix
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset] #creating a new image window that will only show our hand with offset for the extra space around edges

        imgCropShape = imgCrop.shape


        aspectRatio = h/w

        #making overlay Height part

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap= math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:(wCal+wGap)] = imgResize

        #Widht part
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:(hCal+hGap), :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)                         #showing the croped image
        cv2.imshow("ImageWhite", imgWhite)                       #showing the croped image with white BG that will be used for machine learning


    cv2.imshow("Image",img)
    key=cv2.waitKey(1)

    if key == ord("s"):                                          #assigning a key to save the img
        counter += 1
        cv2.imwrite(f'{folder}\image_{counter}.jpg',imgWhite)
        print(counter)


