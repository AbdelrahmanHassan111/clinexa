import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import only the ROI measurement function to test
try:
    from mri_models import extract_roi_measurements
    print("Successfully imported ROI measurement function")
except ImportError as e:
    print(f"Error importing function: {e}")
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

def test_roi_measurements(image_path=None):
    """Test ROI measurements extraction on an image"""
    if image_path is None or not os.path.exists(image_path):
        image_path = create_test_image()
    
    print(f"\nExtracting ROI measurements from: {image_path}")
    
    # Test with different prediction categories
    prediction_categories = ["Normal", "Mild", "Moderate"]
    
    for category in prediction_categories:
        print(f"\nTesting with prediction category: {category}")
        measurements = extract_roi_measurements(image_path, prediction=category)
        
        if measurements:
            print(f"Extracted {len(measurements)} measurements")
            
            # Display key brain region measurements
            key_regions = [
                'hippocampus_left', 'hippocampus_right', 'hippocampus_total',
                'entorhinal_left', 'entorhinal_right', 'entorhinal_total',
                'lateral_ventricles', 'whole_brain'
            ]
            
            print("Key Brain Region Measurements:")
            for region in key_regions:
                if region in measurements:
                    print(f"  {region}: {measurements[region]:.2f} mm³")
            
            # Calculate key ratios
            if all(k in measurements for k in ['hippocampus_total', 'total_intracranial_volume']):
                hipp_ratio = measurements['hippocampus_total'] / measurements['total_intracranial_volume'] * 100
                print(f"  Hippocampus to ICV ratio: {hipp_ratio:.4f}%")
                
            if all(k in measurements for k in ['lateral_ventricles', 'total_intracranial_volume']):
                vent_ratio = measurements['lateral_ventricles'] / measurements['total_intracranial_volume'] * 100
                print(f"  Ventricle to ICV ratio: {vent_ratio:.4f}%")
                
            # Check for asymmetry
            if all(k in measurements for k in ['hippocampus_left', 'hippocampus_right']):
                asymmetry = abs(measurements['hippocampus_left'] - measurements['hippocampus_right']) / (measurements['hippocampus_left'] + measurements['hippocampus_right']) * 100
                print(f"  Hippocampal asymmetry: {asymmetry:.2f}%")
        else:
            print("Failed to extract measurements")
    
    # Create a visualization of measurements
    try:
        # Get measurements for visualization 
        measurements = extract_roi_measurements(image_path, prediction="Moderate")
        
        if measurements:
            # Create a bar chart of key regions
            plt.figure(figsize=(12, 6))
            regions = ['hippocampus_left', 'hippocampus_right', 'entorhinal_left', 'entorhinal_right']
            values = [measurements.get(r, 0) for r in regions]
            
            # Add normal range indicators
            normal_ranges = {
                'hippocampus_left': (3000, 3500),
                'hippocampus_right': (3100, 3600),
                'entorhinal_left': (1800, 2200),
                'entorhinal_right': (1900, 2300)
            }
            
            # Create bars
            bars = plt.bar(regions, values, color=['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e'])
            
            # Add normal range indicators
            for i, region in enumerate(regions):
                if region in normal_ranges:
                    plt.plot([i-0.2, i+0.2], [normal_ranges[region][0], normal_ranges[region][0]], 'g--', linewidth=1)
                    plt.plot([i-0.2, i+0.2], [normal_ranges[region][1], normal_ranges[region][1]], 'g--', linewidth=1)
            
            # Format chart
            plt.ylabel('Volume (mm³)')
            plt.title('Key Brain Region Measurements')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the chart
            output_dir = "test_results"
            os.makedirs(output_dir, exist_ok=True)
            chart_path = os.path.join(output_dir, "brain_measurements.png")
            plt.savefig(chart_path)
            print(f"\nSaved measurement visualization to {chart_path}")
            
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # If a command line argument is provided, use it as the image path
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    test_roi_measurements(image_path) 