"""
Test script for the DHEIVER/Alzheimer-MRI model
This script can be used to verify the model works correctly with a sample image.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Hugging Face Alzheimer's MRI model")
    parser.add_argument("--image", type=str, help="Path to MRI image to test", required=False)
    args = parser.parse_args()
    
    # Try to import transformers
    try:
        from transformers import pipeline
        print("Successfully imported transformers library")
    except ImportError:
        print("ERROR: transformers library not found. Please install with:")
        print("pip install transformers[torch]")
        sys.exit(1)
    
    # Try to import torch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("WARNING: PyTorch not found or CUDA not available.")
        print("The model will run on CPU which might be slower.")
    
    # Initialize the pipeline
    try:
        print("Loading Hugging Face model...")
        pipe = pipeline("image-classification", model="DHEIVER/Alzheimer-MRI")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)
    
    # Get image path
    image_path = args.image
    if not image_path:
        print("No image specified, looking for sample images...")
        # Try to find a sample image in common locations
        sample_dirs = ["temp_uploads", "uploads", "images", "test_images", "."]
        sample_found = False
        
        for directory in sample_dirs:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                        image_path = os.path.join(directory, file)
                        print(f"Found sample image: {image_path}")
                        sample_found = True
                        break
            if sample_found:
                break
        
        if not sample_found:
            print("ERROR: No sample image found and no image path provided.")
            print("Please provide an image path with --image or add images to one of these directories:")
            print(", ".join(sample_dirs))
            sys.exit(1)
    
    # Load and display the image
    try:
        print(f"Loading image from {image_path}...")
        # Load with PIL for display
        img_pil = Image.open(image_path)
        plt.figure(figsize=(5, 5))
        plt.imshow(np.array(img_pil), cmap='gray' if img_pil.mode == 'L' else None)
        plt.title("Input MRI Image")
        plt.axis('off')
        plt.show(block=False)
    except Exception as e:
        print(f"ERROR loading image: {e}")
        sys.exit(1)
    
    # Run the model
    try:
        print("Running prediction...")
        result = pipe(image_path)
        
        print("\n==== PREDICTION RESULTS ====")
        for i, pred in enumerate(result):
            print(f"Class {i+1}: {pred['label']} ({pred['score']*100:.2f}%)")
        
        # Display the top prediction
        top_prediction = result[0]
        print(f"\nTOP PREDICTION: {top_prediction['label']} with {top_prediction['score']*100:.2f}% confidence")
        
        # Keep the plot window open until a key is pressed
        print("\nScript completed successfully. Close the image window to exit.")
        plt.show()
        
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 