import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

print("Current directory contents:")
print(os.listdir('.'))

try:
    model = tf.keras.models.load_model('Object_Recognition_DIYA1.keras')
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    exit()

def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    if img_array is None:
        print("Image preprocessing failed.")
        return None, None

    try:
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        return class_idx, confidence
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None, None

if __name__ == "__main__":
    img_path = '/home/user/Downloads/img1.jpg'
    if not os.path.exists(img_path):
        print(f"Image file not found: {img_path}")
        exit()

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'car']

    class_idx, confidence = predict_image(img_path)
    if class_idx is not None:
        print(f'Predicted class index: {class_idx}')
        print(f'Predicted class name: {class_names[class_idx]}')
        print(f'Confidence: {confidence:.4f}')
        img = image.load_img(img_path)
        plt.imshow(img)
        plt.title(f'Predicted class: {class_names[class_idx]}')
        plt.show()
    else:
        print("Prediction failed.")
