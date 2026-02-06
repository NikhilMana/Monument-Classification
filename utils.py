"""
Utility Functions
Helper functions for plotting, saving results, and preprocessing
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import RESULTS_DIR, IMG_SIZE


def plot_training_history(history, save_path=None):
    """
    Plot training history including loss and accuracy curves.
    
    Args:
        history: Keras training history object or dict
        save_path: Optional path to save the plot
    """
    # Handle both History object and dict
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Plot training & validation loss
    axes[0, 0].plot(history_dict['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history_dict['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot training & validation accuracy
    axes[0, 1].plot(history_dict['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(history_dict['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot top-3 accuracy if available
    if 'top_3_accuracy' in history_dict:
        axes[1, 0].plot(history_dict['top_3_accuracy'], label='Training Top-3', linewidth=2)
        axes[1, 0].plot(history_dict['val_top_3_accuracy'], label='Validation Top-3', linewidth=2)
        axes[1, 0].set_title('Top-3 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot top-5 accuracy if available
    if 'top_5_accuracy' in history_dict:
        axes[1, 1].plot(history_dict['top_5_accuracy'], label='Training Top-5', linewidth=2)
        axes[1, 1].plot(history_dict['val_top_5_accuracy'], label='Validation Top-5', linewidth=2)
        axes[1, 1].set_title('Top-5 Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None, normalize=True):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Optional path to save the plot
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(16, 14))
    
    # Use a custom colormap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_sample_predictions(images, true_labels, predictions, class_names, 
                            num_samples=16, save_path=None):
    """
    Plot sample images with their predictions.
    
    Args:
        images: Array of images
        true_labels: True class indices
        predictions: Predicted probabilities
        class_names: List of class names
        num_samples: Number of samples to plot
        save_path: Optional path to save the plot
    """
    num_samples = min(num_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Display image
        img = images[i]
        if img.max() > 1:
            img = img / 255.0
        ax.imshow(img)
        
        # Get prediction
        pred_idx = np.argmax(predictions[i])
        pred_prob = predictions[i][pred_idx]
        true_idx = true_labels[i] if isinstance(true_labels[i], (int, np.integer)) else np.argmax(true_labels[i])
        
        pred_name = class_names[pred_idx]
        true_name = class_names[true_idx]
        
        # Set title color based on correctness
        color = 'green' if pred_idx == true_idx else 'red'
        
        ax.set_title(
            f'True: {true_name}\nPred: {pred_name}\nConf: {pred_prob:.2%}',
            fontsize=9,
            color=color
        )
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Sample predictions saved to {save_path}")
    
    plt.show()


def plot_class_distribution(class_counts, title="Class Distribution", save_path=None):
    """
    Plot bar chart of class distribution.
    
    Args:
        class_counts: Dictionary of class names to counts
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(14, 6))
    
    names = list(class_counts.keys())
    counts = list(class_counts.values())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    
    bars = plt.bar(range(len(names)), counts, color=colors)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.xlabel('Monument Class')
    plt.ylabel('Number of Images')
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(count),
            ha='center',
            va='bottom',
            fontsize=8
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Class distribution plot saved to {save_path}")
    
    plt.show()


def save_results(results_dict, filename=None):
    """
    Save evaluation results to a text file.
    
    Args:
        results_dict: Dictionary of results to save
        filename: Optional filename (without extension)
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}"
    
    filepath = os.path.join(RESULTS_DIR, f"{filename}.txt")
    
    with open(filepath, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MONUMENT CLASSIFICATION RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for key, value in results_dict.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}:\n{value}\n\n")
    
    print(f"💾 Results saved to {filepath}")
    return filepath


def get_timestamp():
    """Get current timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    # Test plotting functions with dummy data
    print("Testing utility functions...")
    
    # Test training history plot
    dummy_history = {
        'loss': [1.0, 0.8, 0.6, 0.4, 0.3],
        'val_loss': [1.1, 0.9, 0.7, 0.5, 0.45],
        'accuracy': [0.3, 0.5, 0.6, 0.7, 0.75],
        'val_accuracy': [0.25, 0.45, 0.55, 0.65, 0.70],
        'top_3_accuracy': [0.5, 0.7, 0.8, 0.85, 0.9],
        'val_top_3_accuracy': [0.45, 0.65, 0.75, 0.8, 0.85],
        'top_5_accuracy': [0.6, 0.8, 0.85, 0.9, 0.95],
        'val_top_5_accuracy': [0.55, 0.75, 0.8, 0.85, 0.9]
    }
    
    print("✅ Utility functions loaded successfully")
