#!/usr/bin/env python
import os

def fix_syntax():
    file_path = "streamlit_app/doctor_view.py"
    
    # Read the file content as lines
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find and fix the lines with escaped quotes
    for i, line in enumerate(lines):
        if "error_msg = results.get(" in line and "\'" in line:
            # Replace escaped single quotes with proper quotes
            fixed_line = line.replace("\'", "'")
            lines[i] = fixed_line
    
    # Write back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Fixed syntax in {file_path}")

if __name__ == "__main__":
    fix_syntax() 