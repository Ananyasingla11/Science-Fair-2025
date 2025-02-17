import os 
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
import cv2

train_data_dir = 'data/Training_data'
test_data_dir = 'data/Validation_data'

train_datagen = ImageDataGenerator (rescale=1. /255)

# test_datagen = ImageDataGenerator(rescale= 1. /255)

#img = cv2.imread("../data/Case 1.jpg")
#imgheight=img.shape[0]
#imgwidth=img.shape[1]
#y = int((imgheight - 2000)/2)
#x = int((imgwidth - 2000)/2)

imgheight = 299
imgwidth = 299

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size = (imgwidth, imgheight),
                                                    batch_size = 43,
                                                    class_mode = 'categorical',
                                                    subset = 'training')

validation_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size = (imgwidth, imgheight),
                                                    batch_size = 10,
                                                    class_mode = 'categorical',
                                                    subset = 'validation')

# validation_generator = train_datagen.flow_from_directory(train_data_dir,
#                                                     target_size = (imgwidth, imgheight),
#                                                     batch_size = 10,
#                                                     class_mode = 'categorical')

base_model = InceptionV3(weights='imagenet', include_top = False, input_shape=(imgwidth,imgheight,3))
for layer in base_model.layers:
    layer.trainable = False

