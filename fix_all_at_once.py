#!/usr/bin/env python
import os
import re
import sys

def fix_indentation_issues():
    filepath = "streamlit_app/doctor_view.py"
    
    # Create a backup
    backup_filepath = filepath + ".bak"
    if not os.path.exists(backup_filepath):
        with open(filepath, 'r', encoding='utf-8') as src:
            with open(backup_filepath, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print(f"Created backup at {backup_filepath}")
    
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix indentation in important blocks
    
    # Fix lines 1250-1315 (the first problematic section)
    content = re.sub(
        r'(.*# Add MRI scan description section\s*\n)' +
        r'(.*st\.markdown\("### Model Interpretation of MRI Scan"\)\s*\n)' +
        r'(.*with st\.spinner.*\s*\n)' +
        r'(.*description = generate_mri_description.*\s*\n)' +
        r'(.*st\.markdown\(description\)\s*\n)',
        r'\1' +
        r'                                    st.markdown("### Model Interpretation of MRI Scan")\n' +
        r'                                    with st.spinner("Generating detailed scan description..."):\n' +
        r'                                        description = generate_mri_description(results, truncated_scan_type)\n' +
        r'                                        st.markdown(description)\n',
        content, flags=re.MULTILINE
    )
    
    # Fix the else block and subsequent indentation
    content = re.sub(
        r'(.*# Add medical record\s*\n)' +
        r'(.*if add_medical_record.*\s*\n)' +
        r'(.*st\.success.*\s*\n)' +
        r'(.*else:\s*\n)' +
        r'(.*# Display error message\s*\n)' +
        r'(.*error_msg =.*\s*\n)' +
        r'(.*st\.error.*\s*\n)',
        r'\1' +
        r'                                            if add_medical_record(patient_id, diagnosis, notes):\n' +
        r'                                                st.success("Results added to medical records")\n' +
        r'                                else:\n' +
        r'                                    # Display error message\n' +
        r'                                    error_msg = results.get(\'error\', \'Unknown error occurred\') if results else \'Failed to process MRI scan\'\n' +
        r'                                    st.error(f"Error processing MRI scan: {error_msg}")\n',
        content, flags=re.MULTILINE
    )
    
    # Fix debugger section
    content = re.sub(
        r'(.*# Add additional debug information button\s*\n)' +
        r'(.*with st\.expander.*\s*\n)',
        r'\1' +
        r'                                with st.expander("Debug Information"):\n',
        content, flags=re.MULTILINE
    )
    
    # Fix col2 section
    content = re.sub(
        r'(.*with col2:\s*\n)',
        r'                            with col2:\n',
        content, flags=re.MULTILINE
    )
    
    # Fix heatmap section
    content = re.sub(
        r'(.*# Display heatmap if available\s*\n)' +
        r'(.*if selected_scan\.get\(\'is_processed\'\).*\s*\n)',
        r'\1' +
        r'                            if selected_scan.get(\'is_processed\') and selected_scan.get(\'heatmap_path\') and os.path.exists(selected_scan[\'heatmap_path\']):\n',
        content, flags=re.MULTILINE
    )
    
    # Fix AI generation section
    content = re.sub(
        r'(\s+""".*\s*\n\s*\n)' +
        r'(\s+try:\s*\n)',
        r'\1' +
        r'            try:\n',
        content, flags=re.MULTILINE
    )
    
    # Fix warning indentation
    content = re.sub(
        r'(.*# This just logs a warning but doesn\'t affect the overall save operation\s*\n)' +
        r'(.*print\(f"Warning:.*\s*\n)' +
        r'(.*st\.warning\(f"Scan saved.*\s*\n)',
        r'\1' +
        r'                print(f"Warning: Saved scan but couldn\'t store prediction: {e}")\n' +
        r'                st.warning(f"Scan saved successfully, but prediction results couldn\'t be stored: {e}")\n',
        content, flags=re.MULTILINE
    )
    
    # Write the fixed content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed all indentation issues in {filepath}")

if __name__ == "__main__":
    fix_indentation_issues() 