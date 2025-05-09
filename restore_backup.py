#!/usr/bin/env python
import os
import re

def main():
    # File paths
    filepath = "streamlit_app/doctor_view.py"
    backup_filepath = filepath + ".bak"
    
    # Restore from backup if it exists
    if os.path.exists(backup_filepath):
        with open(backup_filepath, 'r', encoding='utf-8') as src:
            original_content = src.read()
            
        # Make specific targeted changes
        # Fix indentation at line 1254
        pattern1 = r'(.*# Add MRI scan description section\s*\n)' + \
                 r'(\s*)st\.markdown\("### Model Interpretation of MRI Scan"\)(\s*\n)' + \
                 r'(\s*)with st\.spinner\("Generating detailed scan description..."\):(\s*\n)' + \
                 r'(\s*)description = generate_mri_description\(results, truncated_scan_type\)(\s*\n)' + \
                 r'(\s*)st\.markdown\(description\)(\s*\n)'
        
        replacement1 = r'\1' + \
                      r'                                    st.markdown("### Model Interpretation of MRI Scan")\n' + \
                      r'                                    with st.spinner("Generating detailed scan description..."):\n' + \
                      r'                                        description = generate_mri_description(results, truncated_scan_type)\n' + \
                      r'                                        st.markdown(description)\n'
        
        fixed_content = re.sub(pattern1, replacement1, original_content, flags=re.DOTALL)
        
        # Fix indentation at around line 1310 (else block)
        pattern2 = r'(.*if add_medical_record.*\s*\n)' + \
                 r'(.*st\.success.*\s*\n)' + \
                 r'(\s*)else:(\s*\n)' + \
                 r'(\s*)# Display error message(\s*\n)' + \
                 r'(\s*)error_msg = .*(\s*\n)' + \
                 r'(\s*)st\.error.*(\s*\n)'
        
        replacement2 = r'\1' + \
                      r'\2' + \
                      r'                                else:\n' + \
                      r'                                    # Display error message\n' + \
                      r'                                    error_msg = results.get(\'error\', \'Unknown error occurred\') if results else \'Failed to process MRI scan\'\n' + \
                      r'                                    st.error(f"Error processing MRI scan: {error_msg}")\n'
        
        fixed_content = re.sub(pattern2, replacement2, fixed_content, flags=re.DOTALL)
        
        # Write the fixed content
        with open(filepath, 'w', encoding='utf-8') as dst:
            dst.write(fixed_content)
            
        print(f"Restored from backup and applied targeted fixes to {filepath}")
    else:
        print(f"Error: Backup file {backup_filepath} not found")
        
if __name__ == "__main__":
    main() 