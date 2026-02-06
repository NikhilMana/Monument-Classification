"""
Configuration file for Indian Monument Classification Pipeline
Optimized settings for CPU Training (Faster iteration)
"""

import os
import kagglehub

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Download dataset and get path
DATASET_PATH = kagglehub.dataset_download("danushkumarv/indian-monuments-image-dataset")
DATA_DIR = os.path.join(DATASET_PATH, "Indian-monuments", "images")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Image dimensions (Reduced to 224 for CPU speed)
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Number of classes (24 Indian monuments)
NUM_CLASSES = 24

# ============================================================================
# TRAINING CONFIGURATION (Optimized for CPU Speed)
# ============================================================================

# Batch size - Reduced for CPU
BATCH_SIZE = 16

# Training epochs (Reduced for faster iteration)
EPOCHS_PHASE1 = 5    # Feature extraction phase
EPOCHS_PHASE2 = 10   # Fine-tuning phase  
TOTAL_EPOCHS = EPOCHS_PHASE1 + EPOCHS_PHASE2

# Learning rates (tuned for EfficientNetV2)
LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE2 = 5e-5

# Validation split from training data
VALIDATION_SPLIT = 0.2

# ============================================================================
# REGULARIZATION & ADVANCED TRAINING
# ============================================================================

DROPOUT_RATE = 0.4

# Label smoothing
LABEL_SMOOTHING = 0.1

# MixUp/CutMix alpha
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0

# ============================================================================
# CALLBACKS CONFIGURATION
# ============================================================================

# Early stopping patience
EARLY_STOPPING_PATIENCE = 5

# Learning rate reduction patience
LR_REDUCE_PATIENCE = 3
LR_REDUCE_FACTOR = 0.5
MIN_LR = 1e-7

# ============================================================================
# PATHS
# ============================================================================

# Model save paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(MODEL_DIR, "monument_classifier_best.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "monument_classifier_final.keras")

# Logs for TensorBoard
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Results directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# CLASS NAMES
# ============================================================================

CLASS_NAMES = sorted(os.listdir(TRAIN_DIR)) if os.path.exists(TRAIN_DIR) else []

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("MONUMENT CLASSIFICATION PIPELINE CONFIGURATION (CPU OPTIMIZED)")
    print("=" * 60)
    print(f"\n📁 Dataset:")
    print(f"   Train Directory: {TRAIN_DIR}")
    print(f"   Test Directory: {TEST_DIR}")
    print(f"   Number of Classes: {NUM_CLASSES}")
    
    print(f"\n🖼️  Image Settings:")
    print(f"   Size: {IMG_HEIGHT}x{IMG_WIDTH} (Reduced for CPU)")
    
    print(f"\n🏋️  Training Settings:")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Phase 1 Epochs: {EPOCHS_PHASE1}")
    print(f"   Phase 2 Epochs: {EPOCHS_PHASE2}")
    
    print(f"\n💾 Model Paths:")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Final Model: {FINAL_MODEL_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    print_config()
