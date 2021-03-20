# disabling tensorflow warnings
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import cv2
from utils import *
from sudoku_solver import *
import operator
from colorama import init, Fore
import time
from copy import *


compute = True
case=28
max_area=0
biggest = None

# using Colorama module for colouring in the shell purposes
RED   = Fore.RED
RESET = Fore.RESET


# video stream loop

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video device")
    print('Could not use webcam')
    quit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
model = launch_model() # CNN digit classifier

while True:
    try: 
        _, frame = cap.read()
        if max_area == 0:
            cv2.imshow("frame", frame) 

        imgThreshold = preprocess(frame)
        #cv2.imshow("Sudoku detector",imgThreshold) # display on camera preprocessed videostream (for debugging purposes)        

        contours = find_contour(imgThreshold)
        biggest, max_area = None, 0
        biggest, max_area = find_biggest_contour(contours)
        
        if biggest is not None:
            print('sudoku detected!')
            # cv2.drawContours(frame,[biggest],0,(0,255,0),2) # display the sudoku borders
            # cv2.imshow('contour',frame)
            cropping = crop_to_sudoku(imgThreshold, biggest)
            if compute is True:
                sudoku_cropped = cropping[0]
                #cv2.imshow("Cropped grid perspective corrected", sudoku_cropped)
                boxes = splitBoxes(sudoku_cropped)
                newboxes = deep_learning_preprocessing(boxes)   
                sudoku_grid = []
                for digit in newboxes:
                    if digit.sum() < 10: # meaning there are very few dark pixels in this cell -> probably an empty cell
                        sudoku_grid.append(0)
                    else:
                        probas = model.predict(digit, batch_size=1)
                        pred = probas.argmax()
                        sudoku_grid.append(pred)
                print('digits recognized!')
                grid = []
                for i in range(9):
                    grid.append(sudoku_grid[9*i:9*(i+1)])
                grid_to_solve = deepcopy(grid) # needed because the solve sudoku function transforms its input, and a shallow copy will be modified too
                solution = solve_sudoku(grid_to_solve)

            if sudoku_validator(solution) is True:
                    print('got a solution!')
                    compute = False
                    for digit in solution:
                        print(*digit) # unpacking the content, for pretty printing in the shell
                    print('\n')

                    # colouring live the empty cells with the solution digit
                    fond = np.zeros(shape=(width, height, 3), dtype=np.float32)
                    for y in range(len(grid)):
                        for x in range(len(grid[y])):
                                if grid[y][x] == 0:
                                    cv2.putText(fond, "{:d}".format(solution[y][x]), (x*case +7, (y+1)*case - 7), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1) #write solution on the empty cell
                    pts1, pts2 = cropping[1], cropping[2]
                    M=cv2.getPerspectiveTransform(pts2, pts1)
                    h, w, c = frame.shape
                    fondP=cv2.warpPerspective(fond, M, (w, h))
                    img2gray=cv2.cvtColor(fondP, cv2.COLOR_BGR2GRAY)
                    ret, mask=cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                    mask=mask.astype('uint8')
                    mask_inv=cv2.bitwise_not(mask)
                    img1_bg=cv2.bitwise_and(frame, frame, mask=mask_inv)
                    img2_fg=cv2.bitwise_and(fondP, fondP, mask=mask).astype('uint8')
                    dst=cv2.add(img1_bg, img2_fg)
                    cv2.imshow("frame", dst)
                    print('displaying solution!')
            else:
                cv2.imshow("frame", frame)
                
        else:
            compute = True
            print("Lost track of the sudoku contour!")
        
        #press q to quit
        key = cv2.waitKey(1)&0xFF
        if key==ord('q'): # press q to quit session
            break
    
    except Exception as e:
        print(f'{RED}Process failed: \n{e}{RESET}')

cap.release()
cv2.destroyAllWindows()