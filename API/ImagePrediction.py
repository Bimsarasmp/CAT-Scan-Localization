import cv2
import os
# Function to load a human anatomy image from a specified file path
def humanImage():
    file = os.getcwd()+"\API\Images\\"
    img = cv2.imread(file+"Human_Anatomy.jpg")    
    return img

# Function to mark a point on the human anatomy image
def markPointInImage(point):
    img = humanImage() # Load the human anatomy image using the defined function
    start = (0,int(4.39*point))
    end = (img.shape[0],int(4.39*point))
    cv2.line(img, start, end, (30, 255, 255), 1)
    cv2.imshow("Human CAT-SCAN Localization", img) # Display the marked image with a window title
    return img # Return the modified image
