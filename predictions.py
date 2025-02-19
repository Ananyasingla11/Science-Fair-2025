import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2

model = load_model('image_classifier_model_with_inceptionv3.h5')



def predict_class(image_path):
    pre_img = cv2.imread(image_path)
    imgheight=pre_img.shape[0]
    imgwidth=pre_img.shape[1]
    y = int((imgheight - 2000)/2)
    x = int((imgwidth - 2000)/2)
    img = image.load_img(image_path, target_size=(299, 299))
    #y = int((imgheight - 2000)/2)
    #x = int((imgwidth - 2000)/2)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.  # Rescale pixel values

    predictions = model.predict(img_array)

    class_labels = ['Healthy', 'Fractured',]  # Define your class labels
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label, predictions[0][predicted_class_index]

which_case = int(input('Which case do you inquire about?' ))
path1 = 'Science-Fair-2025\\Data\\Validation_data\\Healthy\\Case '+str(which_case)+'.jpg'
predicted_class, confidence = predict_class(path1)
print("Predicted class:", predicted_class)
print("Confidence:", confidence)
