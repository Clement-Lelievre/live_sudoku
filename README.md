![sudoku_live_solver](https://raw.githubusercontent.com/Clement-Lelievre/live_sudoku/master/sudoku_live_solver.JPG)

This project aims at creating an app whereby the user can scan a sudoku grid via his webcam and quickly view the solution

I used mainly the following libraries:
-TensorFlow for digit recognition, on an image dataset I created (~400k images)
-Threading to speed up the code
-Opencv for the image preprocessing

Below is a breakdown of the tasks:

-a website able to open user's webcam  
-interact with live stream --> OpenCV python library:
   -image pre-processing (blurring, adaptive thresholding, warp correction etc.)
   -contour finding
   -image cropping
-train a neural network on MNIST database
-because the abovementioned model performed below par, I had to create my own bespoke, printed digits dataset (~400 000 images, with random noise added)
-train a neural network on the abovementioned image dataset
-perform live digit image recognition to transform user grid's photo into a Python, list of lists object 
-pass the abovementioned Python list to a sudoku solver function 
-ideally parallelize the sudoku solver to accelerate solving 
-pretty print the solution
-iterate over the whole process without lag

Credits to Le Wagon (my coding school) and to L42project for helping out on some parts of this project.
