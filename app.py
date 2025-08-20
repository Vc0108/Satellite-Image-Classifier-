import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# Import the pre-trained MobileNetV2 model and its preprocessing functions
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Satellite-Classify AI",
    page_icon="🛰️",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- MODEL AND CLASS NAMES LOADING ---
@st.cache_resource
def load_models():
    """Load the custom satellite classifier and the validation model."""
    try:
        satellite_model = tf.keras.models.load_model('satellite_classifier_model.h5')
        # Load MobileNetV2 trained on ImageNet for validation
        validation_model = MobileNetV2(weights='imagenet')
        return satellite_model, validation_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_class_names():
    """Load the class names for the satellite model from a JSON file."""
    try:
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        return None

satellite_model, validation_model = load_models()
class_names = load_class_names()

# --- HELPER FUNCTIONS ---
def preprocess_for_satellite_model(image):
    """Preprocess the image for the custom satellite model."""
    img = image.resize((128, 128))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.cast(img_array, tf.float32)
    return img_array

def validate_image_type(image, model):
    """
    Validate if the uploaded image is likely a satellite/aerial image using MobileNetV2.
    Returns True if valid, False otherwise.
    """
    # Keywords that suggest a valid satellite/aerial image
    VALID_KEYWORDS = ['coast', 'seashore', 'promontory', 'volcano', 'mountain', 'landscape', 'aerial', 'earth']
    
    # Resize to 224x224 for MobileNetV2
    img = image.resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    
    # Get top 5 predictions
    predictions = model.predict(img_preprocessed)
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    
    st.write("---")
    st.write("**Image Content Analysis (by Gatekeeper AI):**")
    
    is_valid = False
    for _, label, prob in decoded_predictions:
        st.write(f"- I see a '{label.replace('_', ' ')}' with {prob:.1%} confidence.")
        if any(keyword in label for keyword in VALID_KEYWORDS):
            is_valid = True
            
    return is_valid

# --- UI COMPONENTS ---
st.title("🛰️ Sat-Classify AI")
st.write(
    "Welcome! Upload a satellite image and our AI will predict its category. "
    "Is it a cloud formation, a vast desert, a lush green area, or a body of water? Let's find out!"
)
st.divider()

uploaded_file = st.file_uploader(
    "Drag and drop an image here or click to upload",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and satellite_model is not None and validation_model is not None and class_names is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    st.image(image, caption='Your Uploaded Image', use_column_width=True)
    
    if st.button("Classify Image"):
        with st.spinner('Performing initial content analysis...'):
            # Step 1: Validate the image type
            is_valid_image = validate_image_type(image, validation_model)

        if is_valid_image:
            st.success("Validation passed! This looks like an aerial/satellite image.")
            with st.spinner('Analyzing satellite data...'):
                # Step 2: If valid, use the specialized satellite model
                processed_image = preprocess_for_satellite_model(image)
                predictions = satellite_model.predict(processed_image)
                score = tf.nn.softmax(predictions[0])
                predicted_class = class_names[np.argmax(score)]
                confidence = 100 * np.max(score)

                st.success(f"### 🎯 Final Prediction: **{predicted_class}**")
                st.info(f"Confidence: **{confidence:.2f}%**")
        else:
            # If not valid, show a warning and stop
            st.warning("Validation Failed! This does not appear to be a satellite or aerial image. Please upload a different picture.")

elif satellite_model is None or class_names is None:
    st.error("Models or class names could not be loaded. Please ensure the files are in the correct directory.")

# --- SIDEBAR ---
st.sidebar.header("About Sat-Classify AI")
st.sidebar.info(
    "This application uses a two-step process:\n\n"
    "1. **Validation:** A pre-trained MobileNetV2 model checks if the upload is a valid aerial image.\n\n"
    "2. **Classification:** A custom-trained CNN classifies the image into categories like desert, water, etc."
)
st.sidebar.success("Project by a curious developer!")
