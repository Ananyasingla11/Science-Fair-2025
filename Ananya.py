import os 

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
import cv2

test_data_dir = 'Science-Fair-2025/Data/Test_data'
train_data_dir = 'Science-Fair-2025/Data/Training_data'
validation_data_dir = 'Science-Fair-2025/Data/Validation_data'

train_datagen = ImageDataGenerator(rescale=1. /255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1. /255)

# test_datagen = ImageDataGenerator(rescale= 1. /255)

# img = cv2.imread("../data/Case 1.jpg")
# imgheight=img.shape[0]
# imgwidth=img.shape[1]
# y = int((imgheight - 2000)/2)
# x = int((imgwidth - 2000)/2)

imgheight = 299
imgwidth = 299

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size = (imgwidth, imgheight),
                                                    batch_size = 37,
                                                    class_mode = 'categorical',
                                                    subset = 'training')


validation_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size = (imgwidth, imgheight),
                                                    batch_size = 6,
                                                    class_mode = 'categorical',
                                                    subset = 'validation')




test_generator = train_datagen.flow_from_directory(test_data_dir,
                                                    target_size = (imgwidth, imgheight),
                                                    batch_size = 10,
                                                    class_mode = 'categorical')


 

base_model = InceptionV3(weights='imagenet', include_top = False, input_shape=(imgwidth,imgheight,3))
for layer in base_model.layers:
    layer.trainable = False


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_generator,
          steps_per_epoch=len(train_generator),
          epochs=10,
          validation_data=validation_generator,
          validation_steps=len(validation_generator))

print("Model evaluation on test data:")
model.evaluate(test_generator, steps=len(test_generator))

model.save('image_classifier_model_with_inceptionv3.h5')

