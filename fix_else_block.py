#!/usr/bin/env python

def main():
    # Define file path
    file_path = "streamlit_app/doctor_view.py"
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Focus specifically on fixing the else block indentation issue
    fixed_lines = []
    in_editing_region = False
    
    # Line numbers around the problematic else block
    start_line = 1305
    end_line = 1325
    
    for i, line in enumerate(lines):
        line_number = i + 1
        
        # If we're in the editing region, handle indentation specially
        if start_line <= line_number <= end_line:
            in_editing_region = True
            
            # Handle the else statement itself
            if line.strip() == "else:":
                fixed_lines.append("                            else:\n")
                continue
                
            # Handle lines inside the else block
            if line_number > 1310 and in_editing_region and line.strip():
                if "Display error message" in line or "error_msg" in line or "Error processing MRI" in line:
                    fixed_lines.append("                                # Display error message\n" if "Display error" in line else 
                                      "                                error_msg = results.get('error', 'Unknown error occurred') if results else 'Failed to process MRI scan'\n" if "error_msg" in line else
                                      "                                st.error(f\"Error processing MRI scan: {error_msg}\")\n")
                    continue
                elif "Still allow saving" in line or "if st.button" in line:
                    fixed_lines.append("                                # Still allow saving the scan without analysis\n" if "Still allow" in line else 
                                      "                                if st.button(\"Save Scan Without Analysis\"):\n")
                    continue
            
        # Add line as-is if no special handling needed
        if not in_editing_region or line_number > end_line:
            fixed_lines.append(line)
        
        # Reset flag when we pass the editing region
        if line_number == end_line:
            in_editing_region = False
    
    # Write back the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed else block indentation in {file_path}")

if __name__ == "__main__":
    main() 