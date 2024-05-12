import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import json

# Load the model
model = tf.keras.models.load_model("./4new.h5")

# Load plant disease info
with open("plant_info.json", "r") as file:
    plant_disease_info = json.load(file)


# Define function to preprocess image for model input
def preprocess_image(img):
    # Resize and normalize image
    # image = Image.open(img)
    image = np.array(Image.open(img).convert("RGB").resize((256, 256)))
    image = np.expand_dims(image, 0)
    return image


def main():
    # Title and description
    st.title("Plant Disease Prediction App")
    st.write("Upload an image of your plant to predict any diseases.")

    # File upload section
    uploaded_file = st.file_uploader("Choose an image...")

    if uploaded_file is not None:
        # image = Image.open(uploaded_file)
        st.image(uploaded_file, caption="Uploaded Image", width=300)

        # Preprocess image
        processed_img = preprocess_image(uploaded_file)

        # Make prediction
        predictions = model.predict(processed_img) if model else "Model not found"
        predicted_class = plant_disease_info[str(np.argmax(predictions[0]))]

        confidence = float(np.max(predictions[0]))

        confidence_percentage = str(int(round(confidence * 100))) + "%"

        # Display prediction results
        print({"data": predicted_class, "confidence": confidence_percentage})
        st.write({"data": predicted_class, "confidence": confidence_percentage})


if __name__ == "__main__":
    main()
