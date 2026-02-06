"""
Test-Time Augmentation (TTA) and Ensemble Inference
Boost accuracy by averaging predictions from multiple augmented views
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import CHECKPOINT_PATH, FINAL_MODEL_PATH, IMG_SIZE, CLASS_NAMES
from model import load_trained_model


def create_tta_transforms():
    """
    Create TTA transforms for test-time augmentation.
    Each transform produces a different view of the input image.
    """
    return [
        # Original
        lambda x: x,
        # Horizontal flip
        lambda x: tf.image.flip_left_right(x),
        # Slight rotations (via cropping and resizing)
        lambda x: tf.image.central_crop(x, 0.9),
        # Brightness variations
        lambda x: tf.image.adjust_brightness(x, 0.1),
        lambda x: tf.image.adjust_brightness(x, -0.1),
        # Contrast variations
        lambda x: tf.image.adjust_contrast(x, 1.1),
    ]


def predict_with_tta(model, image, num_augmentations=5):
    """
    Predict with Test-Time Augmentation.
    
    Args:
        model: Trained Keras model
        image: Input image tensor (batch_size, H, W, 3)
        num_augmentations: Number of augmented views to average
        
    Returns:
        Averaged predictions
    """
    transforms = create_tta_transforms()[:num_augmentations]
    predictions = []
    
    for transform in transforms:
        augmented = transform(image)
        # Resize back to original size if needed
        if augmented.shape[1:3] != IMG_SIZE:
            augmented = tf.image.resize(augmented, IMG_SIZE)
        pred = model.predict(augmented, verbose=0)
        predictions.append(pred)
    
    # Average all predictions
    mean_pred = np.mean(predictions, axis=0)
    return mean_pred


def predict_single_image(model, image_path, class_names=None, use_tta=True, top_k=5):
    """
    Predict class for a single image with optional TTA.
    
    Args:
        model: Trained Keras model
        image_path: Path to image file
        class_names: List of class names
        use_tta: Whether to use test-time augmentation
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with predictions
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    print(f"\n🔮 Predicting: {image_path}")
    
    # Load and preprocess image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.expand_dims(img, 0)
    
    # Get predictions
    if use_tta:
        print("   Using Test-Time Augmentation (5 views)")
        predictions = predict_with_tta(model, img, num_augmentations=5)[0]
    else:
        predictions = model.predict(img, verbose=0)[0]
    
    # Get top-k predictions
    top_indices = np.argsort(predictions)[::-1][:top_k]
    
    results = {
        'predictions': [],
        'top_class': class_names[top_indices[0]],
        'top_confidence': float(predictions[top_indices[0]])
    }
    
    print(f"\n📊 Top {top_k} Predictions:")
    print("-" * 45)
    
    for i, idx in enumerate(top_indices, 1):
        class_name = class_names[idx]
        confidence = predictions[idx]
        bar = "█" * int(confidence * 25)
        print(f"{i}. {class_name:<25} {confidence:>6.2%} {bar}")
        results['predictions'].append({
            'class': class_name,
            'confidence': float(confidence)
        })
    
    return results


def ensemble_predict(model_paths, image_path, class_names=None, use_tta=False):
    """
    Ensemble prediction from multiple models.
    
    Args:
        model_paths: List of paths to model files
        image_path: Path to image file
        class_names: List of class names
        use_tta: Whether to use TTA on each model
        
    Returns:
        Averaged predictions from all models
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    print(f"\n🎯 Ensemble Prediction with {len(model_paths)} models")
    
    # Load image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.expand_dims(img, 0)
    
    all_predictions = []
    
    for i, model_path in enumerate(model_paths):
        print(f"   Loading model {i+1}: {os.path.basename(model_path)}")
        model = load_trained_model(model_path)
        
        if use_tta:
            pred = predict_with_tta(model, img)[0]
        else:
            pred = model.predict(img, verbose=0)[0]
        
        all_predictions.append(pred)
        
        # Clear model from memory
        del model
        keras.backend.clear_session()
    
    # Average ensemble predictions
    ensemble_pred = np.mean(all_predictions, axis=0)
    top_idx = np.argmax(ensemble_pred)
    
    print(f"\n✅ Ensemble Result: {class_names[top_idx]} ({ensemble_pred[top_idx]:.2%})")
    
    return ensemble_pred, class_names[top_idx], ensemble_pred[top_idx]


def main():
    """Demo TTA inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict with Test-Time Augmentation')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default=None, help='Path to model file')
    parser.add_argument('--no-tta', action='store_true', help='Disable TTA')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions')
    args = parser.parse_args()
    
    # Load model
    model_path = args.model
    if model_path is None:
        model_path = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else FINAL_MODEL_PATH
    
    model = load_trained_model(model_path)
    
    # Make prediction
    results = predict_single_image(
        model, 
        args.image, 
        use_tta=not args.no_tta,
        top_k=args.top_k
    )
    
    return results


if __name__ == "__main__":
    main()
