import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import the MRI processing functions
try:
    from mri_models import (
        process_mri_with_cnn, 
        process_mri_with_swin, 
        process_mri_with_huggingface,
        apply_colormap_to_heatmap,
        extract_roi_measurements
    )
    print("Successfully imported MRI model functions")
except ImportError as e:
    print(f"Error importing MRI functions: {e}")
    exit(1)

def create_test_image():
    """Create a simulated brain MRI image for testing"""
    print("Creating test MRI image...")
    # Create image directory
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a simulated brain-like image
    size = 224
    x, y = np.meshgrid(np.linspace(-3, 3, size), np.linspace(-3, 3, size))
    d = np.sqrt(x*x + y*y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
    
    # Add some random noise and structures
    np.random.seed(42)  # For reproducibility
    
    # Create a brain-like structure with ventricles, etc.
    brain = g + 0.1*np.random.rand(size, size)
    
    # Add "ventricles" in the center
    vent_mask = (np.abs(x) < 0.5) & (np.abs(y) < 0.5)
    brain[vent_mask] *= 0.7  # Darker in ventricle region
    
    # Add "cortical folding" with sine waves
    folding = 0.1 * np.sin(d * 6) * np.exp(-d)
    brain += folding
    
    # Normalize and convert to uint8
    brain = (brain - brain.min()) / (brain.max() - brain.min())
    brain = (brain * 255).astype(np.uint8)
    
    # Convert to RGB
    brain_rgb = cv2.cvtColor(brain, cv2.COLOR_GRAY2RGB)
    
    # Save the image
    test_image_path = os.path.join(test_dir, "test_brain_mri.png")
    cv2.imwrite(test_image_path, brain_rgb)
    print(f"Test image saved to {test_image_path}")
    
    return test_image_path

def test_mri_processing(image_path=None):
    """Test all MRI processing models on a given image or test image"""
    if image_path is None or not os.path.exists(image_path):
        image_path = create_test_image()
    
    print(f"\nTesting MRI models with image: {image_path}")
    
    # Create output directory for results
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Test CNN model
    print("\n1. Testing CNN model...")
    try:
        cnn_results = process_mri_with_cnn(image_path)
        results['cnn'] = cnn_results
        print(f"CNN Prediction: {cnn_results.get('prediction', 'N/A')}")
        print(f"CNN Confidence: {cnn_results.get('confidence', 0):.2f}")
        print(f"CNN Heatmap: {cnn_results.get('heatmap_path', 'N/A')}")
    except Exception as e:
        print(f"Error testing CNN model: {e}")
        import traceback
        traceback.print_exc()
    
    # Test SWIN model
    print("\n2. Testing SWIN model...")
    try:
        swin_results = process_mri_with_swin(image_path)
        results['swin'] = swin_results
        print(f"SWIN Prediction: {swin_results.get('prediction', 'N/A')}")
        print(f"SWIN Confidence: {swin_results.get('confidence', 0):.2f}")
        print(f"SWIN Heatmap: {swin_results.get('heatmap_path', 'N/A')}")
    except Exception as e:
        print(f"Error testing SWIN model: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Hugging Face model
    print("\n3. Testing Hugging Face model...")
    try:
        hf_results = process_mri_with_huggingface(image_path)
        results['huggingface'] = hf_results
        print(f"Hugging Face Prediction: {hf_results.get('prediction', 'N/A')}")
        print(f"Hugging Face Confidence: {hf_results.get('confidence', 0):.2f}")
        print(f"Hugging Face Heatmap: {hf_results.get('heatmap_path', 'N/A')}")
    except Exception as e:
        print(f"Error testing Hugging Face model: {e}")
        import traceback
        traceback.print_exc()
    
    # Test ROI measurements
    print("\n4. Testing ROI measurements extraction...")
    try:
        measurements = extract_roi_measurements(image_path)
        results['measurements'] = measurements
        print(f"ROI Measurements: {len(measurements) if measurements else 0} measurements extracted")
        
        # Display a few key measurements
        if measurements:
            print("\nKey Brain Region Measurements:")
            key_regions = [
                'hippocampus_total', 
                'entorhinal_total', 
                'lateral_ventricles',
                'whole_brain'
            ]
            for region in key_regions:
                if region in measurements:
                    print(f"  {region}: {measurements[region]:.2f} mm³")
    except Exception as e:
        print(f"Error testing ROI measurements: {e}")
        import traceback
        traceback.print_exc()
    
    # Create a comparison visualization
    try:
        print("\nCreating comparison visualization...")
        # Load original image
        original_img = cv2.imread(image_path)
        if original_img is None:
            original_img = np.array(Image.open(image_path))
        
        # Get heatmap paths
        heatmap_paths = [
            results.get('cnn', {}).get('heatmap_path'),
            results.get('swin', {}).get('heatmap_path'),
            results.get('huggingface', {}).get('heatmap_path')
        ]
        
        # Filter out None values
        heatmap_paths = [p for p in heatmap_paths if p and os.path.exists(p)]
        
        # Create comparison image if heatmaps exist
        if heatmap_paths:
            plt.figure(figsize=(15, 5))
            
            # Show original
            plt.subplot(1, len(heatmap_paths) + 1, 1)
            plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            plt.title("Original MRI")
            plt.axis('off')
            
            # Show heatmaps
            for i, path in enumerate(heatmap_paths):
                heatmap_img = cv2.imread(path)
                plt.subplot(1, len(heatmap_paths) + 1, i + 2)
                plt.imshow(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB))
                model_name = ['CNN', 'SWIN', 'Hugging Face'][i] if i < 3 else f"Model {i+1}"
                plt.title(f"{model_name} Heatmap")
                plt.axis('off')
            
            # Save comparison
            comparison_path = os.path.join(output_dir, "mri_models_comparison.png")
            plt.tight_layout()
            plt.savefig(comparison_path)
            print(f"Comparison visualization saved to {comparison_path}")
    except Exception as e:
        print(f"Error creating comparison visualization: {e}")
        import traceback
        traceback.print_exc()
    
    return results

if __name__ == "__main__":
    # If a command line argument is provided, use it as the image path
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    test_mri_processing(image_path) 