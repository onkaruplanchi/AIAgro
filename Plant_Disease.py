import streamlit as st, base64
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os

# Paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "plant_disease_prediction_model.h5")
class_index_path = os.path.join(working_dir, "class_indices.json")

# Load model and class indices
model = tf.keras.models.load_model(model_path)
with open(class_index_path, "r") as f:
    class_indices = json.load(f)

# Reverse class index map
inv_class_indices = {v: k for k, v in class_indices.items()}

# Dictionary mapping disease classes to their cures and medicines
disease_cures = {
    "Apple___Apple_scab": {
        "cure": "Apply fungicide and remove infected leaves to control Apple scab.",
        "medicine": "Fungicides like Captan, Myclobutanil, or Thiophanate-methyl."
    },
    "Apple___Black_rot": {
        "cure": "Prune infected twigs and apply fungicide to manage Black rot in apples.",
        "medicine": "Fungicides like Chlorothalonil or Thiophanate-methyl."
    },
    "Apple___Cedar_apple_rust": {
        "cure": "Remove cedar trees near apple orchards and apply fungicide to control Cedar apple rust.",
        "medicine": "Fungicides like Myclobutanil, Propiconazole, or Thiophanate-methyl."
    },
    "Apple___healthy": {
        "cure": "No specific cure needed for healthy apple trees.",
        "medicine": ""
    },
    "Blueberry___healthy": {
        "cure": "No specific cure needed for healthy blueberry plants.",
        "medicine": ""
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "cure": "Apply fungicide and improve air circulation to manage Powdery mildew in cherries.",
        "medicine": "Fungicides like Sulfur, Potassium bicarbonate, or Azoxystrobin."
    },
    "Cherry_(including_sour)___healthy": {
        "cure": "No specific cure needed for healthy cherry trees.",
        "medicine": ""
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "cure": "Rotate crops and apply fungicide to control Cercospora leaf spot in corn.",
        "medicine": "Fungicides like Azoxystrobin, Pyraclostrobin, or Chlorothalonil."
    },
    "Corn_(maize)___Common_rust_": {
        "cure": "Plant resistant varieties and apply fungicide to manage Common rust in corn.",
        "medicine": "Fungicides like Propiconazole or Tebuconazole."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "cure": "Rotate crops and apply fungicide to control Northern Leaf Blight in corn.",
        "medicine": "Fungicides like Chlorothalonil or Azoxystrobin."
    },
    "Corn_(maize)___healthy": {
        "cure": "No specific cure needed for healthy corn plants.",
        "medicine": ""
    },
    "Grape___Black_rot": {
        "cure": "Prune infected canes and apply fungicide to manage Black rot in grapes.",
        "medicine": "Fungicides like Myclobutanil, Boscalid, or Captan."
    },
    "Grape___Esca_(Black_Measles)": {
        "cure": "Prune infected wood and apply fungicide to control Esca in grapes.",
        "medicine": "Fungicides like Propiconazole, Fludioxonil, or Phosphorous acid."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "cure": "Apply fungicide and prune infected leaves to manage Leaf blight in grapes.",
        "medicine": "Fungicides like Copper sulfate, Mancozeb, or Myclobutanil."
    },
    "Grape___healthy": {
        "cure": "No specific cure needed for healthy grape vines.",
        "medicine": ""
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "cure": "Remove infected trees and control insect vectors to manage Citrus greening in oranges.",
        "medicine": "Systemic antibiotics like Tetracycline or Penicillin."
    },
    "Peach___Bacterial_spot": {
        "cure": "Apply copper sprays and prune infected branches to manage Bacterial spot in peaches.",
        "medicine": "Copper-based fungicides like Copper hydroxide or Copper oxychloride."
    },
    "Peach___healthy": {
        "cure": "No specific cure needed for healthy peach trees.",
        "medicine": ""
    },
    "Pepper,_bell___Bacterial_spot": {
        "cure": "Apply copper-based fungicides and rotate crops to control Bacterial spot in bell peppers.",
        "medicine": "Copper-based fungicides like Copper hydroxide or Copper oxychloride."
    },
    "Pepper,_bell___healthy": {
        "cure": "No specific cure needed for healthy bell pepper plants.",
        "medicine": ""
    },
    "Potato___Early_blight": {
        "cure": "Remove infected leaves and apply fungicide to control Early blight in potatoes.",
        "medicine": "Fungicides like Chlorothalonil or Mancozeb."
    },
    "Potato___Late_blight": {
        "cure": "Remove infected leaves and apply fungicide to manage Late blight in potatoes.",
        "medicine": "Fungicides like Phosphorous acid or Azoxystrobin."
    },
    "Potato___healthy": {
        "cure": "No specific cure needed for healthy potato plants.",
        "medicine": ""
    },
    "Raspberry___healthy": {
        "cure": "No specific cure needed for healthy raspberry plants.",
        "medicine": ""
    },
    "Soybean___healthy": {
        "cure": "No specific cure needed for healthy soybean plants.",
        "medicine": ""
    },
    "Squash___Powdery_mildew": {
        "cure": "Apply fungicide and improve air circulation to manage Powdery mildew in squash.",
        "medicine": "Fungicides like Sulfur, Potassium bicarbonate, or Myclobutanil."
    },
    "Strawberry___Leaf_scorch": {
        "cure": "Apply fungicide and improve soil drainage to manage Leaf scorch in strawberries.",
        "medicine": "Fungicides like Captan, Propiconazole, or Fenhexamid."
    },
    "Strawberry___healthy": {
        "cure": "No specific cure needed for healthy strawberry plants.",
        "medicine": ""
    },
    "Tomato___Bacterial_spot": {
        "cure": "Apply copper-based fungicides and rotate crops to control Bacterial spot in tomatoes.",
        "medicine": "Copper-based fungicides like Copper hydroxide or Copper oxychloride."
    },
    "Tomato___Early_blight": {
        "cure": "Remove infected leaves and apply fungicide to manage Early blight in tomatoes.",
        "medicine": "Fungicides like Chlorothalonil or Mancozeb."
    },
    "Tomato___Late_blight": {
        "cure": "Remove infected leaves and apply fungicide to control Late blight in tomatoes.",
        "medicine": "Fungicides like Phosphorous acid or Azoxystrobin."
    },
    "Tomato___Leaf_Mold": {
        "cure": "Improve air circulation and remove infected leaves to manage Leaf Mold in tomatoes.",
        "medicine": "Fungicides like Chlorothalonil or Mancozeb."
    },
    "Tomato___Septoria_leaf_spot": {
        "cure": "Apply fungicide and remove infected leaves to control Septoria leaf spot in tomatoes.",
        "medicine": "Fungicides like Azoxystrobin or Propiconazole."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "cure": "Apply miticide and improve air circulation to manage Spider mites in tomatoes.",
        "medicine": "Miticides like Abamectin or Spiromesifen."
    },
    "Tomato___Target_Spot": {
        "cure": "Remove infected leaves and apply fungicide to manage Target Spot in tomatoes.",
        "medicine": "Fungicides like Chlorothalonil or Mancozeb."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "cure": "Control whiteflies and remove infected plants to manage Tomato Yellow Leaf Curl Virus.",
        "medicine": "Systemic insecticides like Imidacloprid or Thiamethoxam."
    },
    "Tomato___Tomato_mosaic_virus": {
        "cure": "Control aphids and remove infected plants to manage Tomato Mosaic Virus.",
        "medicine": "Systemic insecticides like Acetamiprid or Thiamethoxam."
    },
    "Tomato___healthy": {
        "cure": "No specific cure needed for healthy tomato plants.",
        "medicine": ""
    }
}

def get_disease_info(class_name):
    return disease_cures.get(class_name, {
        "cure": "No specific treatment found.",
        "medicine": "Consult an expert."
    })

def preprocess_image(image, target_size=(224, 224)):
    image = Image.open(image).resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# UI


def app():
    st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Open Sans;', sans-serif;
    }

    .stApp {

        background-image: url("https://png.pngtree.com/thumb_back/fh260/background/20240725/pngtree-the-concept-of-new-farming-or-smart-farming-agricultural-technology-image_15917909.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }

    img {
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}


    .overlay-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 1rem auto;
        max-width: 900px;
    }

    .title {
        font-size: 36px;
        text-align: center;
        background-color: #032e16;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 2px 2px 12px rgba(0,0,0,0.3);
        font-family: Open Sans;
    }

    .info-box {
    background-color: #e8f5e9;
    border-left: 6px solid #2e7d32;
    color: #1b1b1b;
    font-weight: 500;
    padding: 15px;
    margin-top: 20px;
    border-radius: 8px;
}


    .warning-box {
    background-color: #fff8dc;  /* slightly darker cream */
    border-left: 6px solid #ff9800;
    color: #333333;  /* darker text for readability */
    font-weight: 500;
    padding: 15px;
    margin-top: 10px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

    
    [data-testid="stNotification"] {
        display: none;
    }
    </style>
    """
    ,
    unsafe_allow_html=True
)
    


    st.markdown('<div class="title">🌿 Plant Disease Classifier</div>', unsafe_allow_html=True)
    
    uploaded_image = st.file_uploader("📷 Upload a plant leaf image", type=["jpg", "jpeg", "png"])
    

    if uploaded_image:
        col1, col2 = st.columns([1, 1.2])

        with col1:
            image = Image.open(uploaded_image)
            
            st.image(image, use_container_width=True)



        with col2:
            if st.button("🔍 Diagnose"):
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.markdown(
                    f'<div class="info-box"><strong>🩺 Prediction:</strong> {prediction}</div>',
                    unsafe_allow_html=True
                )

                info = get_disease_info(prediction)

                st.markdown(
                    f'<div class="info-box"><strong>🌿 Cure:</strong> {info["cure"]}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="warning-box"><strong>💊 Recommended Medicine:</strong> {info["medicine"]}</div>',
                    unsafe_allow_html=True
                )




