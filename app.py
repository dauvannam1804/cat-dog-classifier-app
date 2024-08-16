import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('model/cat_dog_classifier_vgg16.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match the input size of the model
    image = np.array(image) / 255.0   # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict the class of the image
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit UI
st.title("Image Classification: Cat vs Dog")

st.write("Upload an image to classify it as a cat or a dog.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    prediction = predict(image)
    
    # Assuming two classes: [0] for 'cat' and [1] for 'dog'
    class_labels = ['Cat', 'Dog']
    predicted_class = np.argmax(prediction)
    
    st.write(f"Prediction: {class_labels[predicted_class]} with probability {np.max(prediction):.2f}")

