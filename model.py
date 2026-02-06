"""
Model Architecture Module
EfficientNetV2-S based transfer learning for high-accuracy monument classification
Optimized for faster training while targeting 95%+ accuracy
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2S

from config import (
    IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES, DROPOUT_RATE,
    LEARNING_RATE_PHASE1, LEARNING_RATE_PHASE2, LABEL_SMOOTHING
)


def create_model(num_classes=NUM_CLASSES, trainable_base=False):
    """
    Create the monument classification model using EfficientNetV2-S transfer learning.
    
    EfficientNetV2-S chosen for optimal balance of:
    - Speed: ~2x faster than EfficientNet-B4
    - Accuracy: State-of-the-art on ImageNet (83.9% top-1)
    - Memory: Efficient for GPU training
    """
    print("🏗️  Building EfficientNetV2-S model...")
    
    # Load EfficientNetV2-S backbone with ImageNet weights
    base_model = EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_preprocessing=True  # Built-in preprocessing for faster pipeline
    )
    
    # Freeze or unfreeze backbone
    base_model.trainable = trainable_base
    
    if trainable_base:
        # Fine-tune only top 30% of layers for speed
        fine_tune_at = int(len(base_model.layers) * 0.7)
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
        print(f"   Fine-tuning mode: {trainable_count}/{len(base_model.layers)} layers trainable")
    else:
        print("   Feature extraction mode: backbone frozen")
    
    # Build the full model with enhanced classification head
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Base model feature extraction
    x = base_model(inputs, training=trainable_base)
    
    # Enhanced classification head with attention
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    # Squeeze-and-Excitation style attention
    se = layers.Dense(256, activation="relu", name="se_dense1")(x)
    se = layers.Dense(x.shape[-1], activation="sigmoid", name="se_dense2")(se)
    x = layers.Multiply(name="se_multiply")([x, se])
    
    # Classification layers with dropout for regularization
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_1")(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4), name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_2")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4), name="dense_2")(x)
    x = layers.Dropout(DROPOUT_RATE * 0.5, name="dropout_3")(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="monument_classifier_v2")
    print(f"✅ EfficientNetV2-S model created with {num_classes} output classes")
    
    return model, base_model


def compile_model(model, learning_rate=LEARNING_RATE_PHASE1, use_mixed_precision=True):
    """
    Compile the model with AdamW optimizer and advanced settings.
    
    Features:
    - AdamW optimizer (weight decay for better generalization)
    - Label smoothing (reduces overconfidence)
    - Mixed precision for 2x faster training on GPU
    """
    if use_mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("✅ Mixed precision enabled (2x faster training)")
        except:
            print("⚠️  Mixed precision not available, using float32")
    
    # AdamW optimizer with weight decay
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-5,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # Categorical crossentropy with label smoothing
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy")
        ]
    )
    print(f"✅ Model compiled with AdamW optimizer, LR: {learning_rate}, Label smoothing: {LABEL_SMOOTHING}")


def unfreeze_model(model, base_model, fine_tune_percent=0.3):
    """
    Unfreeze top layers of the backbone for fine-tuning.
    
    Args:
        model: Full Keras model
        base_model: EfficientNetV2 backbone
        fine_tune_percent: Percentage of layers to unfreeze (default 30% for speed)
    """
    base_model.trainable = True
    fine_tune_at = int(len(base_model.layers) * (1 - fine_tune_percent))
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Keep BatchNorm layers frozen for stable training
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    
    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"🔓 Unfroze {trainable_count} layers ({fine_tune_percent*100:.0f}%) for fine-tuning")
    
    # Recompile with lower learning rate
    compile_model(model, learning_rate=LEARNING_RATE_PHASE2, use_mixed_precision=False)


def get_model_summary(model):
    """Print model summary."""
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    
    trainable = sum([keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable = sum([keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    print(f"Model: {model.name}")
    print(f"Total params: {trainable + non_trainable:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")


def load_trained_model(model_path):
    """Load a trained model from disk."""
    print(f"📂 Loading model from {model_path}")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully")
    return model


if __name__ == "__main__":
    print("Testing EfficientNetV2-S model creation...")
    model, base_model = create_model(trainable_base=False)
    compile_model(model)
    get_model_summary(model)
    
    import numpy as np
    test_input = np.random.rand(1, IMG_HEIGHT, IMG_WIDTH, 3).astype(np.float32)
    output = model.predict(test_input, verbose=0)
    print(f"\n🧪 Test prediction shape: {output.shape}")
    print(f"   Sum of probabilities: {output.sum():.4f}")
