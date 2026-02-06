"""
Evaluation Script - Model evaluation with metrics and visualizations
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score

from config import CHECKPOINT_PATH, FINAL_MODEL_PATH, RESULTS_DIR
from data_loader import load_dataset
from model import load_trained_model
from utils import plot_confusion_matrix, plot_sample_predictions, save_results


def get_predictions(model, dataset):
    """Get model predictions on a dataset."""
    all_images, all_labels, all_preds = [], [], []
    
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())
        all_preds.append(preds)
    
    return np.concatenate(all_images), np.concatenate(all_labels), np.concatenate(all_preds)


def calculate_metrics(y_true, y_pred, class_names):
    """Calculate classification metrics."""
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)
    
    return {
        'accuracy': accuracy_score(y_true_idx, y_pred_idx),
        'top_3_accuracy': top_k_accuracy_score(y_true_idx, y_pred, k=3),
        'top_5_accuracy': top_k_accuracy_score(y_true_idx, y_pred, k=5),
        'classification_report': classification_report(y_true_idx, y_pred_idx, target_names=class_names, digits=4),
        'confusion_matrix': confusion_matrix(y_true_idx, y_pred_idx),
        'per_class_accuracy': dict(zip(class_names, confusion_matrix(y_true_idx, y_pred_idx).diagonal() / confusion_matrix(y_true_idx, y_pred_idx).sum(axis=1)))
    }


def evaluate_model(model_path=None):
    """Main evaluation function."""
    print("\n🔍 MONUMENT CLASSIFIER EVALUATION\n")
    
    if model_path is None:
        model_path = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else FINAL_MODEL_PATH
    
    model = load_trained_model(model_path)
    _, _, test_ds, class_names = load_dataset(augment_train=False)
    
    images, labels, predictions = get_predictions(model, test_ds)
    metrics = calculate_metrics(labels, predictions, class_names)
    
    print(f"Top-1 Accuracy: {metrics['accuracy']:.4f}")
    print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")
    print(f"\n{metrics['classification_report']}")
    
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, 
                         save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_sample_predictions(images, labels, predictions, class_names, 
                           num_samples=16, save_path=os.path.join(RESULTS_DIR, "sample_predictions.png"))
    
    return metrics


if __name__ == "__main__":
    evaluate_model()
