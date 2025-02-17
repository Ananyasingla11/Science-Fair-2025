#%matplotlib inline

# General Libs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline
# import sklearn.metrics
# import random
# import tensorflow as tf
# import matplotlib.pyplot as plt
import os
# import io
# import glob
# import scipy.misc

import pandas as pd
# from six import BytesIO
# from PIL import Image, ImageDraw, ImageFont
# import shutil
# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras import layers
# from tensorflow.keras import Model
# import matplotlib
# from tensorflow.keras.optimizers import RMSprop
# import os
# import zipfile
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import matplotlib.image as mpimg
# from matplotlib.ticker import FormatStrFormatter
# from tensorflow.keras.utils import plot_model
import cv2
import numpy as np
import csv


LEARNING_RATE = 0.0001
labels = ["Fractured", "Healthy", "Fractured-Old"]
IMAGE_HEIGHT, IMAGE_WIDTH = 299, 299

#cropping image, not sure if it works....
img = cv2.imread("../data/Case 1.jpg")
imgheight=img.shape[0]
imgwidth=img.shape[1]
y = int((imgheight - 2000)/2)
x = int((imgwidth - 2000)/2)
crop = img[x:2000, y:2000]
cv2.imshow('original', img) 
cv2.imshow('cropped', crop) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

pre_trained_weights = '../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = InceptionV3(weights=pre_trained_weights, include_top=False, input_shape=(x, y, 3))

x = base_model.output
x = Flatten()(x)
predictions = Dense(196, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freezing pretrained layers
for layer in base_model.layers:
    layer.trainable=False
    
#optimizer = Adam(lr=learning_rate)
#optimizer = keras.optimizers.RMSprop(lr=0.0001)
#model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)  # Add validation split

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_data_dir = '/data'
batch_size = 53
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(imgwidth, imgheight),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    color_mode='rgb',  # Ensure images are loaded in RGB format
                                                    subset='training')  # Specify 'training' subset

validation_generator = train_datagen.flow_from_directory(train_data_dir,
                                                         target_size=(imgwidth, imgheight),
                                                         batch_size=batch_size,
                                                         class_mode='categorical',
                                                         color_mode='rgb',  # Ensure images are loaded in RGB format
                                                         subset='validation')  # Specify 'validation' subset


from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_generator.classes), 
            train_generator.classes)
class_weights

epochs = 10

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='model.h5',
        monitor='val_loss', save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=50,verbose=1)
]

history = model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // BATCH_SIZE,
        epochs=epochs,
        callbacks = callbacks_list,
        validation_data=validation_generator,
        verbose = 1,
        validation_steps=nb_validation_samples // BATCH_SIZE,
        class_weight = class_weights)


# def word_finder(searchstring):
#     csv_file = pd.read_csv('Rib - Sheet1.csv')
#     for searchstring in csv_file:
#         if searchstring in csv_file:
#             print(searchstring[1])
#             return searchstring[1]


# def health_word_finder(searchstring):
#     df = pd.read_csv('Rib - Sheet1.csv', dtype=str, header=None)
#     print(df.loc[df[0] == searchstring, 1])
#     return df.loc[df[0] == searchstring, 1]
# #health_word_finder("Case 1")




def word_finder_health(file_name):
    with open('src\Rib - Sheet1.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, line in enumerate(reader):
            picture_name = line[0]
            if file_name == picture_name:
                print(line[1])
                return(line[1])


def path_remover(filename):
    name_with_extension = os.path.basename(filename)
    name_without_extension = os.path.splitext(name_with_extension)[0]
    print(name_without_extension)
    return name_without_extension

#print(word_finder_case_name('Case 53'))


def ask_human():
    image_num = input (str('Which case do you ponder about? '))
    filename = 'Case ' + image_num
    health_status = word_finder_health(filename)
    return(health_status)

def robot_lookup(image_path):
    filename = path_remover(image_path)
    health_status = word_finder_health(filename)
    return(health_status)


#robot_lookup('../data/Case 34.jpg')
ask_human()