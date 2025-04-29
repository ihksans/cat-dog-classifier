import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ======== Load model (caching di Hugging Face) ========
@st.cache_resource
def load_model():
    model_path = "cat_dog_classifier.h5"
    return tf.keras.models.load_model(model_path)

model = load_model()

# ======== Konfigurasi ========
IMG_SIZE = (160, 160)
CLASS_NAMES = ["Cat", "Dog"]

# ======== UI ========
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.title("🐾 Cat vs Dog Classifier")
st.write("Upload gambar kucing atau anjing, dan model akan menebaknya.")

# Upload file
uploaded_file = st.file_uploader("Upload gambar (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocess
    img_resized = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = CLASS_NAMES[1] if prediction >= 0.5 else CLASS_NAMES[0]
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    st.markdown(f"### 🧠 Prediksi: **{label}**")
    st.markdown(f"🎯 Kepercayaan: `{confidence:.2%}`")
