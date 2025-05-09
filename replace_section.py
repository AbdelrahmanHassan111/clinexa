#!/usr/bin/env python

def main():
    # Define file paths
    original_file = "streamlit_app/doctor_view.py"
    fixed_section_file = "doctor_view_fixed.py"
    
    # Read the fixed section
    with open(fixed_section_file, 'r', encoding='utf-8') as f:
        fixed_section = f.read()
    
    # Read the original file
    with open(original_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Define the problematic range
    start_line = 1300  # 1-indexed
    end_line = 1315    # 1-indexed
    
    # Create the new content by replacing the problematic section
    new_lines = lines[:start_line-1]  # Keep everything before the problematic section
    new_lines.append(fixed_section)   # Add our fixed section
    new_lines.extend(lines[end_line:])  # Add everything after the problematic section
    
    # Write the new content back to the file
    with open(original_file, 'w', encoding='utf-8') as f:
        if isinstance(new_lines[0], list):
            f.writelines(new_lines)
        else:
            for line in new_lines:
                if isinstance(line, str):
                    f.write(line)
                else:
                    f.write("".join(line))
    
    print(f"Replaced problematic section in {original_file}")

if __name__ == "__main__":
    main() 