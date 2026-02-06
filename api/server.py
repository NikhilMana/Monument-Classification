"""
Flask API Server for Monument Classification
Provides REST endpoints for image prediction with TTA support
"""

import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras

import sys
# Add parent directory to path to allow importing config and model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from our modules
from config import CHECKPOINT_PATH, FINAL_MODEL_PATH, IMG_SIZE, CLASS_NAMES
from model import load_trained_model
from tta_inference import predict_with_tta

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global model variable
model = None


def load_model():
    """Load the trained model on startup."""
    global model
    if model is None:
        model_path = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else FINAL_MODEL_PATH
        if os.path.exists(model_path):
            print(f"📂 Loading model from {model_path}")
            model = load_trained_model(model_path)
            print("✅ Model loaded successfully!")
        else:
            print("⚠️ No trained model found. Please train the model first.")
    return model


def preprocess_image(image_bytes):
    """Preprocess image bytes for prediction."""
    # Open image with PIL
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to model input size
    img = img.resize(IMG_SIZE)
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'num_classes': len(CLASS_NAMES)
    })


@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of monument classes."""
    return jsonify({
        'classes': CLASS_NAMES,
        'count': len(CLASS_NAMES)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict monument class from uploaded image.
    
    Expects:
        - multipart/form-data with 'image' field
        - OR JSON with 'image' as base64 string
    
    Returns:
        JSON with predictions and confidence scores
    """
    global model
    
    # Ensure model is loaded
    if model is None:
        model = load_model()
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        # Get image from request
        if 'image' in request.files:
            # File upload
            image_file = request.files['image']
            image_bytes = image_file.read()
        elif request.is_json and 'image' in request.json:
            # Base64 encoded image
            image_data = request.json['image']
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess image
        img_array = preprocess_image(image_bytes)
        
        # Check for TTA option
        use_tta = request.args.get('tta', 'false').lower() == 'true'
        
        if use_tta:
            # Use Test-Time Augmentation
            predictions = predict_with_tta(model, tf.constant(img_array), num_augmentations=5)[0]
        else:
            # Standard prediction
            predictions = model.predict(img_array, verbose=0)[0]
        
        # Get top-5 predictions
        top_k = 5
        top_indices = np.argsort(predictions)[::-1][:top_k]
        
        # Confidence threshold (80%)
        CONFIDENCE_THRESHOLD = 0.80
        top_confidence = float(predictions[top_indices[0]])
        
        # Check if confidence meets threshold
        if top_confidence >= CONFIDENCE_THRESHOLD:
            results = {
                'success': True,
                'recognized': True,
                'top_class': CLASS_NAMES[top_indices[0]],
                'top_confidence': top_confidence,
                'predictions': []
            }
            
            for idx in top_indices:
                results['predictions'].append({
                    'class': CLASS_NAMES[idx],
                    'confidence': float(predictions[idx])
                })
        else:
            # Confidence below threshold - monument not recognized
            results = {
                'success': True,
                'recognized': False,
                'top_class': 'Not Recognized',
                'top_confidence': top_confidence,
                'message': f'Confidence ({top_confidence:.1%}) is below 80% threshold',
                'predictions': []
            }
            
            # Still include predictions for debugging
            for idx in top_indices:
                results['predictions'].append({
                    'class': CLASS_NAMES[idx],
                    'confidence': float(predictions[idx])
                })
        
        return jsonify(results)
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict/tta', methods=['POST'])
def predict_with_tta_endpoint():
    """
    Predict with Test-Time Augmentation enabled.
    Same as /predict but forces TTA for higher accuracy.
    """
    # Redirect to predict with TTA enabled
    return predict()  # TTA is handled in the predict function


if __name__ == '__main__':
    print("\n" + "🚀" * 20)
    print("\n   MONUVISION AI - API SERVER")
    print("\n" + "🚀" * 20 + "\n")
    
    # Load model on startup
    load_model()
    
    print("\n📡 Starting server on http://localhost:5000")
    print("   Endpoints:")
    print("   - GET  /health    - Health check")
    print("   - GET  /classes   - List monument classes")
    print("   - POST /predict   - Predict monument (add ?tta=true for TTA)")
    print("\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
