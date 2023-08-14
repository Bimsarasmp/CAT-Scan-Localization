import cv2

def humanImage():
    file = r"D:/Sem-2/3014 - Introduction to Artificial Intelligence/Project/"
    img = cv2.imread(file+"Human_Anatomy.jpg")    
    return img

def markPointInImage(point):
    img = humanImage()
    start = (0,int(4.39*point))
    end = (img.shape[0],int(4.39*point))
    #start = (img.shape[1]//2,0)
    #end = (img.shape[1]//2,img.shape[0])
    cv2.line(img, start, end, (30, 255, 255), 1)
    cv2.imshow("Human CAT-SCAN Localization", img)
    imgFile = cv2.imwrite("CatScan.jpg",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return imgFile