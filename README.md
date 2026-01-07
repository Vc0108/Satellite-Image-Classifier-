## Deployment Link:
https://satellite-image-classify.streamlit.app/

# 🛰️ Satellite Imagery Classifier

This repository contains a **Satellite Image Classification** system that utilizes a Deep Learning model to categorize remote sensing imagery into distinct land-use or terrain classes.

## 🚀 Overview
The project provides an automated pipeline for classifying satellite photos (such as forests, urban areas, or water bodies) by processing them through a trained Convolutional Neural Network (CNN).

## 📂 Repository Structure
* **`app.py`**: The main application entry point (Streamlit/Flask) for the user interface and model inference.
* **`satellite_classifier_model.h5`**: The pre-trained Keras/TensorFlow model file.
* **`class_names.json`**: Mapping file that connects model output indices to human-readable labels.
* **`requirements.txt`**: List of Python dependencies required to run the project.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow / Keras
* **Interface:** Streamlit
* **Image Processing:** OpenCV / PIL
* **Environment:** Python 3.x

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Vc0108/satellite-classifier.git](https://github.com/Vc0108/satellite-classifier.git)
   cd satellite-classifier
