from tensorflow.keras import datasets
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping


####################
# old, do not reuse#
####################
'''This script is run only once in order to train a deep learning model to classify digits. The model is trained using the MNIST dataset.   
The model is then saved in the models directory of this project for later use.
'''

# pre-processing steps
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data(path="mnist.npz")
X_train = X_train / 255.
X_test = X_test / 255.
X_train = expand_dims(X_train, axis=-1)
X_test = expand_dims(X_test, axis=-1)
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

# creating a CNN model

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

model = initialize_model()

# fit the model on the training data

es = EarlyStopping(patience=2)

model.fit(X_train, y_train_cat,
          batch_size=32,
          epochs=10,
          validation_split=0.3,
          callbacks=[es])

# assess model performance on the test set 

res = model.evaluate(X_test, y_test_cat, verbose=0)
print(f'The accuracy on the test set is of {res[1]*100:.3f}')


# save model for later use

models.save_model(model, "../models/digit_classification.h5") 
json_string = model.to_json()
