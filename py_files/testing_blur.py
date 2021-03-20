import cv2
from utils import *
from sudoku_solver import *
import operator
from colorama import init, Fore
import time

cap = cv2.VideoCapture(0)
coeff = 9
photo_saved = False
c = True
flag = 0
marge=0
case=28#+2*marge
taille_grille=9*case


# using Colorama module for colouring in the shell purposes
RED   = Fore.RED
RESET = Fore.RESET


# video stream loop
#model = launch_model()
while True:
    try:
        ret, frame = cap.read()
        def preprocess(img):
                #img = cv2.resize(img,(width,height)) #resizing
            imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # gray scale the inputted sudoku image
            imgBlur = cv2.GaussianBlur(imgGray,(5,5),0) # add gaussian Gaussian Blur
            imgThreshold = cv2.adaptiveThreshold(imgBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,coeff,2) # apply adaptive threshold
            return imgThreshold
        imgThreshold = preprocess(frame)
        #cv2.imshow("Sudoku detector",imgThreshold) # display on camera preprocessed videostream (for debugging purposes)
        
        ##keys for testing image thresholding during live stream
        key = cv2.waitKey(1)&0xFF
        if key==ord('q'): # to quit
            break
        if key==ord('p'): # increase thresholding
            coeff = min(21,coeff+2)
        if key==ord('m'):  # decrease thresholding
            coeff = max(3,coeff-2)

        
    
        cv2.imshow('Sudoku',imgThreshold) 

      
        


    
    except Exception as e:
        print(f'{RED}Process failed: \n{e}{RESET}')

cap.release()
cv2.destroyAllWindows()