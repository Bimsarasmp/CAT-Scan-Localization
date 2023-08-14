import cv2
import os

def humanImage():
    file = os.getcwd()+"\API\Images\\"
    img = cv2.imread(file+"Human_Anatomy.jpg")    
    return img

def markPointInImage(point):
    img = humanImage()
    start = (0,int(4.39*point))
    end = (img.shape[0],int(4.39*point))
    cv2.line(img, start, end, (30, 255, 255), 1)
    cv2.imshow("Human CAT-SCAN Localization", img)
    return img