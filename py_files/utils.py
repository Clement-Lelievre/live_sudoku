# Here are the main functions used throughout the process (mainly image pre-processing, sudoku solving mainly)

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import math
import operator
from datetime import datetime
import time

now = str(datetime.now()).replace(' ','_').replace(':','_').replace('.','')
photoname = '../images_from_users/cropped_'+ now +'.jpg'


width = 28*9 # the 28 comes from MNIST dataset sample size, and the 9 is the number of boxes in a sudoku grid
height = 28*9
coeff = 9

# checking the user's photo to prevent malware

# saving the user's photo in sudoku_images directory

# image pre-processing

def save_image(path, image):
    return cv2.imwrite(path, image)


def path_to_image(path):
    img = cv2.imread(path) 
    return img

def preprocess(img):
    #img = cv2.resize(img,(width,height)) #resizing
    imgGray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY) # gray scale the inputted sudoku image
    imgBlur = cv2.GaussianBlur(imgGray,(9,9),0) # add gaussian Gaussian Blur
    imgThreshold = cv2.adaptiveThreshold(imgBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,coeff,2) # apply adaptive threshold
    imgThreshold = cv2.bitwise_not(imgThreshold) #invert colours  
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
    output = cv2.dilate(imgThreshold, kernel) # enlarge gridlines
    return output


# image contour-finder

def find_contour(img):
    #imgBigcontour = img.copy() # for display and debugging purposes
    cont, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find the contours from the thresholded image
    #cv2.drawContours(img,contours,-1,(0,255,0),3) # write contours in green (hence the (0,255,0)--> RGB)
    return cont

def find_biggest_contour(contours):
    biggest, max_area = None, 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 30000: # this excludes surfaces such as 9 cells high by 6 cells wide (9*28*6*28)
            peri = cv2.arcLength(contour, True)
            polygone = cv2.approxPolyDP(contour, 0.02*peri, True)
            if area > max_area and len(polygone) == 4: # len 4 ensures we are dealing with a rectangle or square (our sudoku grid)
                biggest = polygone
                max_area = area
    return biggest, max_area

def reorder(points):
    '''reorders the 4 corners of the sudoku grid, to make sure they are read in the right order'''
    points = points.reshape((4,2))
    newpoints = np.zeros((4,1,2), dtype = np.int32)
    add = points.sum(1)
    newpoints[0] = points[np.argmin(add)]
    newpoints[3] = points[np.argmax(add)]
    diff = np.diff(points,axis=1)
    newpoints[1] = points[np.argmin(diff)]
    newpoints[2] = points[np.argmax(diff)]
    return newpoints

# cropping image to get only the sudoku grid
def crop_to_sudoku(img, points):
    '''crops the image to keep only the sudoku grid (assumed to be the largest 4-sided polygon on the image), and corrects the perspective'''
    points=np.vstack(points).squeeze()
    points=sorted(points, key=operator.itemgetter(1))
    if points[0][0]<points[1][0]:
        if points[3][0]<points[2][0]:
            pts1=np.float32([points[0], points[1], points[3], points[2]])
        else:
            pts1=np.float32([points[0], points[1], points[2], points[3]])
    else:
        if points[3][0]<points[2][0]:
            pts1=np.float32([points[1], points[0], points[3], points[2]])
        else:
            pts1=np.float32([points[1], points[0], points[2], points[3]])
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    M=cv2.getPerspectiveTransform(pts1, pts2)
    grille=cv2.warpPerspective(img, M, (width, height)) # perspective correction
    return [grille, pts1, pts2]

# splitting the cropped image into the 81 boxes
def splitBoxes(img):
    '''split a sudoku grid image into 81 cells. The order is : lines, left to right. '''
    img = np.array(img)
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

# deep learning image preprocessing

def deep_learning_preprocessing(list_of_img):
    boxes_to_predict = []
    for img in list_of_img:
        img =  img.astype('float32')
        img = img/255.
        for i in [0,1,2,3,24,25,26,27]:
            img[i] = np.zeros((1,28)) # accounting for the grids within the sudoku: removing the dark pixels making up the inner lines
        for row in img:
            for i in [0,1,2,3,24,25,26,27]:
                row[i] = 0 # accounting for the grids within the sudoku: removing the dark pixels making up the inner lines
        img = np.reshape(img,(1,28,28,1))
        boxes_to_predict.append(img)
    return boxes_to_predict

def deep_learning_preprocessing_variant(list_of_img):
    '''Additional image pre-processing in order for new digits (outside of MNIST) to be predicted well. '''
    boxes_to_predict = []
    # below functions will be used to center the digit in the box
    def getBestShift(img):
        cy,cx = ndimage.measurements.center_of_mass(img)
        rows,cols = img.shape
        shiftx = np.round(cols/2.0-cx).astype(int)
        shifty = np.round(rows/2.0-cy).astype(int)
        return shiftx,shifty
    
    def shift(img,sx,sy):
        rows,cols = img.shape
        M = np.float32([[1,0,sx],[0,1,sy]])
        shifted = cv2.warpAffine(img,M,(cols,rows))
        return shifted
    
    # remove side pixels to focus on the digit
    for img in list_of_img:
        img = img/255.
        while np.sum(img[0]) == 0:
            img = img[1:]
        while np.sum(img[:,0]) == 0:
            img = np.delete(img,0,1)
        while np.sum(img[-1]) == 0:
            img = img[:-1]
        while np.sum(img[:,-1]) == 0:
            img = np.delete(img,-1,1)
        rows,cols = img.shape
        if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            img = cv2.resize(img, (cols,rows))
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            img = cv2.resize(img, (cols, rows))
        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
        img = np.lib.pad(img,(rowsPadding,colsPadding),'constant')
        # center the digit
        shiftx, shifty = getBestShift(img)
        shifted = shift(img,shiftx,shifty)
        img = shifted
        img = img.reshape(1,28,28,1)       
        boxes_to_predict.append(img)
    return boxes_to_predict

# using a classification model in order to categorize the digits on the grid

def launch_model():
    model = load_model('../models/printed_digits_CNN_classifier.h5')
    return model

# solving the sudoku grid
 # see dedicated module

