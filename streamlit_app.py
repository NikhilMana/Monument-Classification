"""
MonuVision AI - Streamlit Deployment
Monument Recognition using EfficientNetV2
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="MonuVision AI | Monument Recognition",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Gen Z aesthetic
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    h1 {
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .tagline {
        text-align: center;
        color: #e0e0e0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .confidence-high {
        color: #4ade80;
        font-weight: bold;
        font-size: 2rem;
    }
    .confidence-medium {
        color: #fbbf24;
        font-weight: bold;
        font-size: 2rem;
    }
    .confidence-low {
        color: #f87171;
        font-weight: bold;
        font-size: 2rem;
    }
    .monument-name {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-item {
        background: rgba(255, 255, 255, 0.05);
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load the trained EfficientNetV2 model"""
    try:
        # Try to load the best model first
        model_path = Path("models/monument_classifier_best.keras")
        if not model_path.exists():
            # Fallback to final model
            model_path = Path("models/monument_classifier_final.keras")
        
        if not model_path.exists():
            st.error("⚠️ Model file not found. Please ensure model exists in 'models/' directory.")
            return None
            
        model = tf.keras.models.load_model(str(model_path))
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_class_names():
    """Load monument class names from config"""
    try:
        import config
        if config.CLASS_NAMES:
            return config.CLASS_NAMES
    except:
        pass
    
    # Fallback class names if config fails
    class_names = [
        "Taj Mahal", "Red Fort", "Qutub Minar", "Gateway of India",
        "India Gate", "Hawa Mahal", "Charminar", "Victoria Memorial",
        "Mysore Palace", "Golden Temple", "Ajanta Caves", "Ellora Caves",
        "Konark Sun Temple", "Hampi", "Fatehpur Sikri", "Sanchi Stupa",
        "Khajuraho", "Mahabalipuram", "Brihadeeswara Temple", "Amer Fort",
        "Lotus Temple", "Akshardham", "Meenakshi Temple", "Jama Masjid"
    ]
    return class_names

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize to model input size (384x384 for EfficientNetV2-S)
    img = image.resize((384, 384))
    img_array = np.array(img)
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict(model, image, class_names):
    """Make prediction on the image"""
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    
    # Get top 5 predictions
    top_indices = np.argsort(predictions[0])[::-1][:5]
    top_predictions = [(class_names[i], float(predictions[0][i]) * 100) for i in top_indices]
    
    return top_predictions

# Header
st.markdown("<h1>🏛️ Monu<span style='background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Vision</span></h1>", unsafe_allow_html=True)
st.markdown("<p class='tagline'>AI-Powered Monument Recognition</p>", unsafe_allow_html=True)

# Load model
model = load_model()
class_names = load_class_names()

if model is not None:
    # File uploader
    uploaded_file = st.file_uploader(
        "Drop your monument image here",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Upload an image of an Indian monument"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Make prediction
        with st.spinner("🔍 Analyzing monument..."):
            predictions = predict(model, image, class_names)
        
        # Display results
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        
        monument_name, confidence = predictions[0]
        
        # Confidence color coding
        if confidence >= 80:
            conf_class = "confidence-high"
        elif confidence >= 50:
            conf_class = "confidence-medium"
        else:
            conf_class = "confidence-low"
        
        st.markdown(f"<p class='monument-name'>{monument_name}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='{conf_class}' style='text-align: center;'>{confidence:.1f}% Confidence</p>", unsafe_allow_html=True)
        
        # Progress bar
        st.progress(confidence / 100)
        
        # Top 5 predictions
        st.markdown("### 📊 Top Predictions")
        for i, (name, conf) in enumerate(predictions, 1):
            st.markdown(f"""
            <div class='prediction-item'>
                <strong>#{i}</strong> {name} - <span style='color: #667eea;'>{conf:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Try again button
        if st.button("🔄 Try Another Image", use_container_width=True):
            st.rerun()
else:
    st.warning("⚠️ Model not loaded. Please check your model file.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #e0e0e0;'>Powered by EfficientNetV2 • Built with ❤️</p>", unsafe_allow_html=True)
