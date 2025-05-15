import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Constants
IMG_SIZE = 128
CLASS_NAMES = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Title
st.title("🌸 Flower Classifier")
st.write("Upload an image of a flower and let the model predict its type.")

# Upload image
uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "png"])


# Prediction function
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)[0]

    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    if confidence > 50:
        st.markdown(f"### 🌼 Prediction: **{predicted_class}**")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
    else:
        st.markdown(f"### 🌼 Prediction: **flower not identified**")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
    # Optional: Show all class probabilities
    st.subheader("🔍 Class Probabilities:")
    for i, prob in enumerate(predictions):
        st.write(f"{CLASS_NAMES[i]}: {prob:.2%}")
