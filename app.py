import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ======== Konfigurasi ========
MODEL_PATH = "cat_dog_classifier.h5"
IMG_SIZE = (160, 160)
CLASS_NAMES = ["Cat", "Dog"]
# =============================

# ======== Load Model ========
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ======== UI ========
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload gambar dan biarkan model menebak apakah itu kucing atau anjing.")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocessing
    img_resized = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = img_array / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = CLASS_NAMES[1] if prediction >= 0.5 else CLASS_NAMES[0]
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    st.markdown(f"### Prediksi: **{label}**")
    st.markdown(f"Kepercayaan: `{confidence:.2%}`")
