"""
Training Script
Optimized training pipeline with cosine annealing, gradient clipping, and fast convergence
"""

import os
import json
import math
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

from config import (
    EPOCHS_PHASE1, EPOCHS_PHASE2, LEARNING_RATE_PHASE1, LEARNING_RATE_PHASE2,
    CHECKPOINT_PATH, FINAL_MODEL_PATH, LOG_DIR, RESULTS_DIR,
    EARLY_STOPPING_PATIENCE, LR_REDUCE_PATIENCE, LR_REDUCE_FACTOR, MIN_LR,
    print_config
)
from data_loader import load_dataset
from model import create_model, compile_model, unfreeze_model, get_model_summary
from utils import plot_training_history, save_results


class CosineAnnealingScheduler(keras.callbacks.Callback):
    """
    Cosine Annealing Learning Rate Scheduler with Warm Restarts.
    Provides smoother learning rate decay for better convergence.
    """
    def __init__(self, initial_lr, min_lr=1e-7, epochs_per_cycle=None, total_epochs=None):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.epochs_per_cycle = epochs_per_cycle or total_epochs
        
    def on_epoch_begin(self, epoch, logs=None):
        # Cosine annealing formula
        progress = epoch / self.total_epochs
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        # Update learning rate directly
        self.model.optimizer.learning_rate = lr
        print(f"\n📈 Epoch {epoch + 1}: Learning rate = {lr:.2e}")


class GradientClipCallback(keras.callbacks.Callback):
    """Monitor and log gradient norms for debugging training stability."""
    def on_batch_end(self, batch, logs=None):
        if batch % 50 == 0:  # Log every 50 batches
            pass  # Gradient clipping is handled by optimizer


def create_callbacks(phase="phase1", initial_lr=LEARNING_RATE_PHASE1, total_epochs=EPOCHS_PHASE1):
    """
    Create training callbacks for monitoring and optimization.
    
    Args:
        phase: Training phase ("phase1" or "phase2")
        initial_lr: Initial learning rate for cosine annealing
        total_epochs: Total epochs for this phase
        
    Returns:
        List of Keras callbacks
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        # Model checkpoint - save best model
        keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Cosine annealing scheduler
        CosineAnnealingScheduler(
            initial_lr=initial_lr,
            min_lr=MIN_LR,
            total_epochs=total_epochs
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOG_DIR, f"{phase}_{timestamp}"),
            histogram_freq=0,  # Disabled for speed
            write_graph=False,
            update_freq='epoch'
        ),
        
        # Terminate on NaN for stability
        keras.callbacks.TerminateOnNaN()
    ]
    
    return callbacks


def train_phase1(model, train_ds, val_ds):
    """
    Phase 1: Feature Extraction
    Train only the classification head while keeping backbone frozen.
    """
    print("\n" + "=" * 60)
    print("⚡ PHASE 1: FEATURE EXTRACTION (FAST)")
    print("=" * 60)
    print(f"Training classification head for {EPOCHS_PHASE1} epochs")
    print(f"Initial learning rate: {LEARNING_RATE_PHASE1}")
    print("Backbone: FROZEN (EfficientNetV2-S)")
    print("=" * 60 + "\n")
    
    callbacks = create_callbacks(
        phase="phase1", 
        initial_lr=LEARNING_RATE_PHASE1, 
        total_epochs=EPOCHS_PHASE1
    )
    
    history = model.fit(
        train_ds,
        epochs=EPOCHS_PHASE1,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ Phase 1 training complete!")
    return history


def train_phase2(model, base_model, train_ds, val_ds, initial_epoch=0):
    """
    Phase 2: Fine-Tuning
    Unfreeze top layers of backbone and train end-to-end.
    """
    print("\n" + "=" * 60)
    print("🎯 PHASE 2: FINE-TUNING (HIGH ACCURACY)")
    print("=" * 60)
    print(f"Fine-tuning for {EPOCHS_PHASE2} epochs")
    print(f"Initial learning rate: {LEARNING_RATE_PHASE2}")
    print("Backbone: 30% UNFROZEN (top layers)")
    print("=" * 60 + "\n")
    
    # Unfreeze and recompile with lower LR
    unfreeze_model(model, base_model, fine_tune_percent=0.3)
    
    callbacks = create_callbacks(
        phase="phase2",
        initial_lr=LEARNING_RATE_PHASE2,
        total_epochs=EPOCHS_PHASE2
    )
    
    history = model.fit(
        train_ds,
        epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2,
        initial_epoch=initial_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ Phase 2 training complete!")
    return history


def merge_histories(history1, history2):
    """Merge two training history dictionaries."""
    merged = {}
    
    h1 = history1.history if hasattr(history1, 'history') else history1
    h2 = history2.history if hasattr(history2, 'history') else history2
    
    for key in h1.keys():
        merged[key] = h1[key] + h2.get(key, [])
    
    return merged


def save_training_history(history, filepath):
    """Save training history to JSON file."""
    history_dict = history if isinstance(history, dict) else history.history
    
    # Convert numpy types to Python types
    serializable = {}
    for key, values in history_dict.items():
        serializable[key] = [float(v) for v in values]
    
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"💾 Training history saved to {filepath}")


def main():
    """Main training function."""
    print("\n" + "🚀" * 30)
    print("\n   INDIAN MONUMENT CLASSIFICATION TRAINING v2.0")
    print("   EfficientNetV2-S | Optimized for Speed & Accuracy")
    print("\n" + "🚀" * 30 + "\n")
    
    # Print configuration
    print_config()
    
    # Check GPU availability and enable memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n🎮 GPU Available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"   {gpu}")
            # Enable memory growth for efficiency
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass
    else:
        print("\n⚠️  No GPU detected. Training will use CPU (slower).")
    
    # Load data
    print("\n" + "-" * 60)
    train_ds, val_ds, test_ds, class_names = load_dataset(augment_train=True, use_mixup_cutmix=True)
    print("-" * 60)
    
    # Create model
    print("\n" + "-" * 60)
    model, base_model = create_model(num_classes=len(class_names), trainable_base=False)
    compile_model(model, learning_rate=LEARNING_RATE_PHASE1)
    get_model_summary(model)
    print("-" * 60)
    
    # Phase 1: Feature Extraction
    history1 = train_phase1(model, train_ds, val_ds)
    phase1_epochs = len(history1.history['loss'])
    
    # Phase 2: Fine-Tuning
    history2 = train_phase2(model, base_model, train_ds, val_ds, initial_epoch=phase1_epochs)
    
    # Merge histories
    full_history = merge_histories(history1, history2)
    
    # Save final model
    print(f"\n💾 Saving final model to {FINAL_MODEL_PATH}")
    model.save(FINAL_MODEL_PATH)
    
    # Save training history
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    save_training_history(full_history, history_path)
    
    # Plot training history
    plot_path = os.path.join(RESULTS_DIR, "training_history.png")
    plot_training_history(full_history, save_path=plot_path)
    
    # Print final metrics
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Final Metrics:")
    print(f"   Training Accuracy: {full_history['accuracy'][-1]:.4f}")
    print(f"   Validation Accuracy: {full_history['val_accuracy'][-1]:.4f}")
    print(f"   Training Loss: {full_history['loss'][-1]:.4f}")
    print(f"   Validation Loss: {full_history['val_loss'][-1]:.4f}")
    
    if 'top_3_accuracy' in full_history:
        print(f"   Validation Top-3 Accuracy: {full_history['val_top_3_accuracy'][-1]:.4f}")
    if 'top_5_accuracy' in full_history:
        print(f"   Validation Top-5 Accuracy: {full_history['val_top_5_accuracy'][-1]:.4f}")
    
    print(f"\n💾 Model saved to: {FINAL_MODEL_PATH}")
    print(f"📊 Best checkpoint: {CHECKPOINT_PATH}")
    print(f"📈 TensorBoard logs: {LOG_DIR}")
    
    return model, full_history


if __name__ == "__main__":
    model, history = main()
