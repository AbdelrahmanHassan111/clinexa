"""
Utility script to download the DHEIVER/Alzheimer-MRI model from Hugging Face
and cache it locally for offline use in the Alzheimer Diagnosis System.
"""

import os
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, pipeline
from pathlib import Path

print("Starting model download script...")

# Set up cache directory
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(model_dir, exist_ok=True)

# Model ID to download
MODEL_ID = "DHEIVER/Alzheimer-MRI"
print(f"Downloading model: {MODEL_ID}")

try:
    # Download and cache the model
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    
    # Create and test the pipeline to ensure it's properly cached
    pipe = pipeline("image-classification", model=MODEL_ID)
    
    # Get the cache location
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    
    print(f"Model downloaded successfully!")
    print(f"Model is cached at: {cache_dir}")
    print(f"To use this model offline, ensure the above directory is preserved.")
    
    # Verify model classes
    if hasattr(model, "config") and hasattr(model.config, "id2label"):
        print("\nModel classification labels:")
        for idx, label in model.config.id2label.items():
            print(f"  Class {idx}: {label}")
    
    print("\nDownload completed successfully. The model will now be available for offline use.")
    
except Exception as e:
    print(f"Error downloading model: {e}")
    import traceback
    traceback.print_exc() 