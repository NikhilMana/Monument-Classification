"""
Script to find corrupted images using TensorFlow's decoder
"""

import os
import tensorflow as tf
from config import TRAIN_DIR, TEST_DIR

def check_images_tf(directory):
    """Find corrupted images using TensorFlow's image decoder."""
    corrupted = []
    checked = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                filepath = os.path.join(root, file)
                checked += 1
                try:
                    raw = tf.io.read_file(filepath)
                    img = tf.image.decode_image(raw, channels=3)
                    _ = img.numpy()  # Force evaluation
                except Exception as e:
                    print(f"❌ Corrupted: {filepath}")
                    print(f"   Error: {str(e)[:80]}")
                    corrupted.append(filepath)
                
                if checked % 200 == 0:
                    print(f"   Checked {checked} images...")
    
    return corrupted, checked


def main():
    print("🔍 Scanning with TensorFlow decoder...\n")
    
    print("📁 Checking training directory...")
    train_corrupted, train_checked = check_images_tf(TRAIN_DIR)
    
    print("\n📁 Checking test directory...")  
    test_corrupted, test_checked = check_images_tf(TEST_DIR)
    
    all_corrupted = train_corrupted + test_corrupted
    
    print(f"\n{'='*60}")
    print(f"Total checked: {train_checked + test_checked}")
    print(f"Corrupted: {len(all_corrupted)}")
    
    if all_corrupted:
        response = input("\nRemove corrupted files? (y/n): ")
        if response.lower() == 'y':
            for f in all_corrupted:
                os.remove(f)
                print(f"   Removed: {f}")
            print(f"\n✅ Removed {len(all_corrupted)} files")


if __name__ == "__main__":
    main()
