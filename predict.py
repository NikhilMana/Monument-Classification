"""
Prediction/Inference Module
Load trained model and make predictions with optional TTA
"""

import os
import argparse
import numpy as np
import tensorflow as tf

from config import CHECKPOINT_PATH, FINAL_MODEL_PATH, IMG_SIZE, CLASS_NAMES
from model import load_trained_model
from data_loader import load_single_image
from tta_inference import predict_with_tta, predict_single_image


def predict_image(model, image_path, class_names, top_k=5, use_tta=False):
    """
    Predict class for a single image.
    
    Args:
        model: Trained Keras model
        image_path: Path to image file
        class_names: List of class names
        top_k: Number of top predictions to return
        use_tta: Whether to use Test-Time Augmentation
    """
    print(f"\n🔮 Predicting: {image_path}")
    
    # Load and preprocess image
    img = load_single_image(image_path)
    
    # Get predictions
    if use_tta:
        print("   Using Test-Time Augmentation (5 views)")
        predictions = predict_with_tta(model, img, num_augmentations=5)[0]
    else:
        predictions = model.predict(img, verbose=0)[0]
    
    # Get top-k predictions
    top_indices = np.argsort(predictions)[::-1][:top_k]
    
    print(f"\n📊 Top {top_k} Predictions:")
    print("-" * 45)
    
    for i, idx in enumerate(top_indices, 1):
        class_name = class_names[idx]
        confidence = predictions[idx]
        bar = "█" * int(confidence * 25)
        print(f"{i}. {class_name:<25} {confidence:>6.2%} {bar}")
    
    return class_names[top_indices[0]], predictions[top_indices[0]]


def predict_batch(model, image_paths, class_names, use_tta=False):
    """Predict classes for multiple images."""
    results = []
    for path in image_paths:
        pred_class, confidence = predict_image(model, path, class_names, use_tta=use_tta)
        results.append({'path': path, 'prediction': pred_class, 'confidence': confidence})
    return results


def main():
    parser = argparse.ArgumentParser(description='Predict monument class from image')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default=None, help='Path to model file')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions')
    parser.add_argument('--tta', action='store_true', help='Use Test-Time Augmentation for higher accuracy')
    args = parser.parse_args()
    
    # Load model
    model_path = args.model
    if model_path is None:
        model_path = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else FINAL_MODEL_PATH
    
    print(f"📂 Loading model from {model_path}")
    model = load_trained_model(model_path)
    
    # Make prediction
    pred_class, confidence = predict_image(
        model, 
        args.image, 
        CLASS_NAMES, 
        top_k=args.top_k,
        use_tta=args.tta
    )
    
    print(f"\n✅ Final Prediction: {pred_class} ({confidence:.2%})")


if __name__ == "__main__":
    main()
