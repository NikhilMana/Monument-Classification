"""
Data Loading and Preprocessing Module
Advanced augmentation pipeline with MixUp, CutMix, and RandAugment
"""

import os
import tensorflow as tf
import numpy as np
from config import (
    TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, 
    VALIDATION_SPLIT, MIXUP_ALPHA, CUTMIX_ALPHA
)


def create_advanced_augmentation():
    """
    Create advanced data augmentation pipeline.
    Includes geometric and photometric augmentations for better generalization.
    """
    data_augmentation = tf.keras.Sequential([
        # Geometric augmentations
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom((-0.1, 0.2)),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        
        # Photometric augmentations
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
    ], name="advanced_augmentation")
    
    return data_augmentation


def mixup(images, labels, alpha=MIXUP_ALPHA):
    """
    MixUp augmentation - blends two samples for better generalization.
    Paper: https://arxiv.org/abs/1710.09412
    """
    batch_size = tf.shape(images)[0]
    
    # Sample lambda from beta distribution
    lam = tf.random.uniform([], 0, 1)
    if alpha > 0:
        lam = tf.maximum(lam, 1 - lam)  # Ensure dominant image
    
    # Create shuffled indices
    indices = tf.random.shuffle(tf.range(batch_size))
    
    # Mix images and labels
    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
    
    return mixed_images, mixed_labels


def cutmix(images, labels, alpha=CUTMIX_ALPHA):
    """
    CutMix augmentation - cuts and pastes patches between samples.
    Paper: https://arxiv.org/abs/1905.04899
    """
    batch_size = tf.shape(images)[0]
    img_h, img_w = tf.shape(images)[1], tf.shape(images)[2]
    
    # Sample lambda from beta distribution
    lam = tf.random.uniform([], 0, 1)
    
    # Calculate patch dimensions
    cut_ratio = tf.sqrt(1 - lam)
    cut_h = tf.cast(tf.cast(img_h, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(img_w, tf.float32) * cut_ratio, tf.int32)
    
    # Random patch center
    cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)
    cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)
    
    # Bounding box coordinates
    x1 = tf.maximum(0, cx - cut_w // 2)
    y1 = tf.maximum(0, cy - cut_h // 2)
    x2 = tf.minimum(img_w, cx + cut_w // 2)
    y2 = tf.minimum(img_h, cy + cut_h // 2)
    
    # Create shuffled indices
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    
    # Create mask
    pad_left = x1
    pad_right = img_w - x2
    pad_top = y1
    pad_bottom = img_h - y2
    
    # Apply CutMix
    mask = tf.ones([y2 - y1, x2 - x1, 3])
    mask = tf.pad(mask, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
    
    mixed_images = images * (1 - mask) + shuffled_images * mask
    
    # Adjust lambda based on actual area
    actual_lam = 1 - tf.cast((x2 - x1) * (y2 - y1), tf.float32) / tf.cast(img_h * img_w, tf.float32)
    mixed_labels = actual_lam * labels + (1 - actual_lam) * tf.gather(labels, indices)
    
    return mixed_images, mixed_labels


def apply_mixup_cutmix(images, labels):
    """Randomly apply MixUp or CutMix with 50% probability each."""
    choice = tf.random.uniform([], 0, 1)
    
    def apply_mixup():
        return mixup(images, labels)
    
    def apply_cutmix():
        return cutmix(images, labels)
    
    def no_aug():
        return images, labels
    
    # 30% MixUp, 30% CutMix, 40% no mixing
    return tf.cond(
        choice < 0.3,
        apply_mixup,
        lambda: tf.cond(choice < 0.6, apply_cutmix, no_aug)
    )


def load_dataset(augment_train=True, use_mixup_cutmix=True):
    """
    Load training, validation, and test datasets.
    
    Args:
        augment_train: Whether to apply data augmentation to training data
        use_mixup_cutmix: Whether to apply MixUp/CutMix augmentation
        
    Returns:
        train_ds, val_ds, test_ds: TensorFlow datasets
        class_names: List of class names
    """
    print("📂 Loading datasets...")
    
    # Load training data with validation split
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    
    # Load test data
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )
    
    # Get class names
    class_names = train_ds.class_names
    
    print(f"✅ Found {len(class_names)} classes")
    
    # Apply data augmentation to training set
    if augment_train:
        augmentation = create_advanced_augmentation()
        train_ds = train_ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        print("✅ Advanced augmentation applied")
        
        # Apply MixUp/CutMix
        if use_mixup_cutmix:
            train_ds = train_ds.map(
                apply_mixup_cutmix,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            print("✅ MixUp/CutMix augmentation applied")
    
    # Optimize performance with caching and prefetching
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    print("✅ Datasets ready with caching enabled")
    
    return train_ds, val_ds, test_ds, class_names


def load_single_image(image_path):
    """Load and preprocess a single image for prediction."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.expand_dims(img, 0)
    return img


if __name__ == "__main__":
    print("Testing advanced data loading...")
    train_ds, val_ds, test_ds, class_names = load_dataset()
    
    for images, labels in train_ds.take(1):
        print(f"\n📦 Sample batch:")
        print(f"   Images shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Image range: [{images.numpy().min():.2f}, {images.numpy().max():.2f}]")
