This project aims at creating a website where the user can scan a sudoku grid via his webcam and quickly get the solved grid.

To do this I need:

-a website able to open user's webcam and take a photo | WEBDEV
-save the photo | WEBDEV
-perform multi-digit image recognition (localisation + identification) to transform user grid's photo into a Python, list of lists object | MACHINE LEARNING
-pass the abovementioned Python list to a sudoku solver function | PURE PYTHON
-ideally parallelize the sudoku solver to accelerate solving | PURE PYTHON
-ideally pretty print the solved grid