import streamlit as st
import os
from PIL import Image

def debug_mri_processing(scan_info):
    """Simple debug function that won't cause syntax errors"""
    st.warning("⚠️ MRI Processing Debug Mode")
    
    # Check file path
    if 'file_path' in scan_info and scan_info['file_path']:
        if os.path.exists(scan_info['file_path']):
            st.success(f"✅ File exists at: {scan_info['file_path']}")
            try:
                img = Image.open(scan_info['file_path'])
                st.image(img, width=300, caption="MRI Image")
                st.success(f"✅ Image loaded successfully. Size: {img.size}")
            except Exception as e:
                st.error(f"❌ Error loading image: {e}")
        else:
            st.error(f"❌ File not found at: {scan_info['file_path']}")
    else:
        st.error("❌ No file path specified")
    
    return True 