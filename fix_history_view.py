#!/usr/bin/env python

def main():
    file_path = "streamlit_app/doctor_view.py"
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Get the line with the if statement to determine its indentation
    for i, line in enumerate(lines):
        if "if add_medical_record(patient_id, diagnosis, notes):" in line:
            if_line_num = i
            if_indentation = len(line) - len(line.lstrip())
            break
    
    # Find the corresponding else line
    else_line_num = None
    for i in range(if_line_num + 1, len(lines)):
        if "else:" in lines[i].strip():
            else_line_num = i
            break
    
    if if_line_num is not None and else_line_num is not None:
        # Create the properly indented else line with the same indentation as the if
        if_spaces = " " * if_indentation
        else_line = f"{if_spaces}else:\n"
        
        # Replace the problematic else line
        lines[else_line_num] = else_line
        
        # Find and fix indentation of lines inside else block
        current_line = else_line_num + 1
        while current_line < len(lines) and (lines[current_line].strip() == "" or len(lines[current_line]) - len(lines[current_line].lstrip()) > if_indentation):
            if lines[current_line].strip() != "":
                # Add 4 more spaces to the if indentation for lines inside the else block
                lines[current_line] = if_spaces + "    " + lines[current_line].lstrip()
            current_line += 1
    
    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Fixed indentation in {file_path}")

if __name__ == "__main__":
    main() 