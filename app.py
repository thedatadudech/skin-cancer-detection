import os
import logging
import numpy as np
from PIL import Image
import streamlit as st
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize PyTorch
TORCH_AVAILABLE = False
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    TORCH_AVAILABLE = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except Exception as e:
    logger.error(f"Failed to initialize PyTorch: {str(e)}")

# Class labels
CLASSES = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
           'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']

from src.preprocessing import preprocess_image
from src.model import load_model

def load_model_for_inference():
    """Load the PyTorch model with error handling"""
    try:
        model_path = 'models/best_model.pth'
        if not os.path.exists('models'):
            os.makedirs('models')
            st.warning("""
            Model directory not found. The model needs to be trained first.
            Please train the model using the training script before using the application.
            """)
            return None

        if not os.path.exists(model_path):
            st.warning("""
            Model file not found. The model needs to be trained first.
            Please make sure to:
            1. Download the HAM10000 dataset
            2. Place it in the data directory
            3. Run the training script (train.py)
            """)
            return None

        model = load_model(model_path)
        model.eval()  # Set to evaluation mode
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Error loading model: {str(e)}")
        return None

# Page config
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide"
)

st.title("Skin Cancer Detection System")

st.markdown("""
This application helps detect various types of skin cancer from dermoscopic images.
Please upload a clear, well-lit image of the skin lesion for analysis.
""")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None

if not TORCH_AVAILABLE:
    st.error("""
    ⚠️ PyTorch initialization failed. Please try refreshing the page.
    If the problem persists, contact technical support.
    """)
else:
    # Load model if not already loaded
    if st.session_state.model is None:
        with st.spinner('Loading model...'):
            st.session_state.model = load_model_for_inference()

    if st.session_state.model is None:
        st.info("""
        ### Getting Started
        To use this application, you need to:
        1. Download the HAM10000 dataset from Harvard Dataverse
        2. Place the dataset in the data directory
        3. Run the training script to generate the model

        The model will be automatically loaded once training is complete.
        """)
    else:
        # Main interface
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)

                if st.button('Analyze Image'):
                    with st.spinner('Processing...'):
                        # Preprocess image
                        image_tensor = preprocess_image(image)

                        # Make prediction
                        with torch.no_grad():
                            outputs = st.session_state.model(image_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            predicted_class = torch.argmax(probabilities, dim=1).item()
                            confidence = probabilities[0][predicted_class].item()

                        st.success('Analysis Complete!')
                        st.write(f"**Predicted Class:** {CLASSES[predicted_class]}")
                        st.write(f"**Confidence:** {confidence:.2%}")

                        # Show probability distribution
                        probs_dict = {class_name: prob.item() for class_name, prob in zip(CLASSES, probabilities[0])}
                        st.bar_chart(probs_dict)

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                logger.error(f"Error processing image: {str(e)}")

st.markdown("""
### About
This system uses deep learning to analyze dermoscopic images and detect various types of skin lesions.
The model has been trained on the HAM10000 dataset and can identify 7 different types of skin conditions.

**Disclaimer:** This tool is for educational purposes only. Always consult a healthcare professional for medical advice.
""")