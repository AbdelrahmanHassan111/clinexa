#!/usr/bin/env python
import os
import re

def fix_indentation():
    file_path = "streamlit_app/doctor_view.py"
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the problematic section using regex
    pattern = r'(.*?if add_medical_record\(patient_id, diagnosis, notes\):\s*\n.*?st\.success\(.*?\)\s*\n)(\s*else:\s*\n)(\s*# Display error message\s*\n)(\s*error_msg = .*?\s*\n)(\s*st\.error\(.*?\)\s*\n)'
    
    # The replacement should have proper indentation after the else:
    replacement = r'\1                                            else:\n                                                # Display error message\n                                                error_msg = results.get(\'error\', \'Unknown error occurred\') if results else \'Failed to process MRI scan\'\n                                                st.error(f"Error processing MRI scan: {error_msg}")\n'
    
    # Apply the replacement
    fixed_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"Fixed indentation in {file_path}")

if __name__ == "__main__":
    fix_indentation() 