import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import joblib as jbl
from PIL import Image

def predict_image(img_):

    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

    img = img_.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0) 

    prediction = model.predict(img)
    class_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    predicted_class = classes[class_index]

    return predicted_class, confidence  

model = tf.keras.models.load_model("./models/best_model.h5")


#### Interface Streamlit ####

st.set_page_config(page_title="BrainScan", layout="centered")
st.title("BrainScan - Détection du Cancer Cérébral par CNN")

uploaded_file = st.file_uploader("Choisis une image", type=['jpg', 'jpeg', 'png','bmp'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.image(img, caption="Image importée", use_container_width=True)

    pred_class, conf = predict_image(img)

    st.text(f"Classe prédite : {pred_class}")
    st.text(f"Score de confiance : {conf:.2f}%")