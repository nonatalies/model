from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import json
from PIL import Image


def find_object(img_path):
    pillow_image = Image.open(img_path)
    img_width, img_height = pillow_image.size

    # Prepare image
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create batch dimension
    img_array = img_array / 255.0  # Normalize pixel values

    # Load the trained model
    model = load_model('./model.keras')

    # Make predictions
    predictions = model.predict(img_array)

    # Assuming the model predicts class probabilities for multiple classes
    predicted_class = np.argmax(predictions)  # Get the index of the class with the highest probability

    # Convert predicted_class to string before checking
    predicted_class_str = str(predicted_class)

    # Load dictionary from JSON
    with open('dictionary_ru.json', 'r', encoding='utf-8') as json_file:
        class_labels = json.load(json_file)

    # Assuming 'predicted_class' contains the index of the predicted class
    if predicted_class_str in class_labels:
        return class_labels.get(predicted_class_str)
    else:
        return None
