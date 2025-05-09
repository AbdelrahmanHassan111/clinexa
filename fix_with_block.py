#!/usr/bin/env python

def main():
    # Define file path
    file_path = "streamlit_app/doctor_view.py"
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Focus specifically on fixing the with block indentation issue
    fixed_lines = []
    for i, line in enumerate(lines):
        # Line numbers are 0-indexed in our program, but 1-indexed in error messages
        line_number = i + 1
        
        # Fix line 1255: with st.spinner indent
        if line_number == 1255 and "with st.spinner" in line:
            # Fix indentation level
            fixed_lines.append("                                    with st.spinner(\"Generating detailed scan description...\"):\n")
            continue
        
        # Fix line 1256-1257: content inside the with block
        if line_number == 1256 and "description" in line:
            # Fix indentation for the with block content
            fixed_lines.append("                                        description = generate_mri_description(results, truncated_scan_type)\n")
            continue
            
        if line_number == 1257 and "st.markdown" in line:
            # Fix indentation for the with block content
            fixed_lines.append("                                        st.markdown(description)\n")
            continue
        
        # Add line as-is if no special handling needed
        fixed_lines.append(line)
    
    # Write back the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed with block indentation in {file_path}")

if __name__ == "__main__":
    main() 