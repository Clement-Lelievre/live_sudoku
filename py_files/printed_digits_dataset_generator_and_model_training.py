from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import os
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from random import randint

nbr=2*42

def noise_image(image, seuil=1.3):
    '''adds noise to the images, to be more realistic for the classifier model '''
    b=np.random.normal(0, 1, (28, 28,1))
    a=image.copy()
    a[b>seuil]=255
    a[b<-seuil]=0
    return a

tab_images, tab_labels= [], []

# step 1 generate the dataset of printed digits (MNIST does not work well for such digits)
for dir in ["C:\Windows\Fonts"]: # this directory depends on the OS
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith("ttf"):
                print(root+"/"+file)
                for i in range(1, 10):
                    for cpt in range(nbr):
                        image=Image.new("L", (28, 28))
                        draw=ImageDraw.Draw(image)
                        font=ImageFont.truetype(root+"/"+file, np.random.randint(26, 32))
                        text="{:d}".format(i)
                        draw.text((np.random.randint(1, 10), np.random.randint(-4, 0)), text, font=font, fill=(255))
                        image=np.array(image).reshape(28, 28, 1)
                        tab_images.append(image)
                        tab_labels.append(np.eye(10)[i])                        
                        image_noised=noise_image(image)
                        tab_images.append(image_noised)
                        tab_labels.append(np.eye(10)[i])
                image=np.zeros((28, 28, 1))
                for cpt in range(3*nbr): # creation of the "zeros", which are the empty cells in the sudoku grid
                    image_m=noise_image(image)
                    tab_images.append(image_noised)
                    tab_labels.append(np.eye(10)[0])
                    
tab_images=np.array(tab_images)
tab_labels=np.array(tab_labels)

tab_images=tab_images/255.

tab_images, tab_labels = shuffle(tab_images, tab_labels)

if False: # Set to True to view generated images
    for i in range(len(tab_images)):
        cv2.imshow('chiffre', tab_images[i].reshape(28, 28, 1))
        print(tab_labels[i], np.argmax(tab_labels[i]))
        if cv2.waitKey()&0xFF==ord('q'): # press q to quit
            break

print("Number of images created in the dataset:", len(tab_images))

# step 2 define train/test datasets
train_images, test_images, train_labels, test_labels = train_test_split(tab_images, tab_labels, test_size=0.2)

# step 3 define a CNN model

def initialize_model():
    model = models.Sequential()

    ### First convolution & max-pooling
    model.add(layers.Conv2D(8, (4,4), input_shape=(28, 28, 1), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    ### Flattening
    model.add(layers.Flatten())

    ### One fully connected
    model.add(layers.Dense(10, activation='relu'))

    ### Last layer (classification with 10 outputs: the digits 0 to 9)
    model.add(layers.Dense(10, activation='softmax'))
    
    ### Model compilation
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    return model

# step 4 instantiates the model and start training it on the training set
model = initialize_model()

es = EarlyStopping(patience=2)

model.fit(train_images, train_labels,
          batch_size=32,
          epochs=10,
          validation_split=0.3,
          callbacks=[es])

# step 5 assess model performance on the test set 

res = model.evaluate(test_images, test_labels, verbose=0)
print(f'The accuracy on the test set is of {res[1]*100:.3f}')

# step 6 save model for later use

models.save_model(model, "../models/printed_digits_CNN_classifier.h5") 
json_string = model.to_json()

