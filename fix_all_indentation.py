#!/usr/bin/env python

def main():
    # Define file path
    file_path = "streamlit_app/doctor_view.py"
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix specific indentation errors
    fixed_lines = []
    for i, line in enumerate(lines):
        # Line numbers are 0-indexed in our program, but 1-indexed in error messages
        line_number = i + 1
        
        # Fix line 1254: MRI scan description section
        if line_number == 1254 and "Model Interpretation of MRI Scan" in line:
            # Fix indentation to match surrounding code (36 spaces)
            fixed_lines.append("                                    " + line.lstrip())
            continue
            
        # Fix line 1255: spinner indentation
        if line_number == 1255 and "with st.spinner" in line:
            # Fix indentation to match surrounding code (36 spaces)
            fixed_lines.append("                                    " + line.lstrip())
            continue
            
        # Fix lines 1310-1315: error message display
        if line_number == 1310 and "else:" in line.strip():
            # Fix indentation for the else block
            fixed_lines.append("                            else:\n")
            continue
            
        if line_number >= 1311 and line_number <= 1315:
            # Fix indentation for lines inside the else block
            fixed_lines.append("                                " + line.lstrip())
            continue
            
        # Fix lines 1333-1336: debug information expander
        if line_number == 1333 and "with st.expander" in line:
            # Fix indentation
            fixed_lines.append("                                " + line.lstrip())
            continue
            
        if line_number >= 1334 and line_number <= 1336 and "st.write" in line:
            # Fix indentation for debug info lines
            fixed_lines.append("                                    " + line.lstrip())
            continue
            
        # Fix lines 1516-1519: display scan image
        if line_number == 1516 and "with col2:" in line:
            # Fix indentation
            fixed_lines.append("                        with col2:\n")
            continue
            
        if line_number >= 1517 and line_number <= 1519:
            # Fix indentation for lines inside the col2 block
            fixed_lines.append("                            " + line.lstrip())
            continue
            
        # Fix lines 1536-1539: display heatmap
        if line_number == 1536 and "if selected_scan.get('is_processed')" in line:
            # Fix indentation
            fixed_lines.append("                            " + line.lstrip())
            continue
            
        if line_number >= 1537 and line_number <= 1539:
            # Fix indentation for lines inside the if block
            fixed_lines.append("                                " + line.lstrip())
            continue
            
        # Fix lines 2315-2318: AI assistant response generation
        if line_number == 2315 and "try:" in line.strip():
            # Fix indentation
            fixed_lines.append("            try:\n")
            continue
            
        if line_number >= 2316 and line_number <= 2318:
            # Fix indentation for lines inside the try block
            fixed_lines.append("                " + line.lstrip())
            continue
            
        # Fix lines 2683-2686: save MRI scan warning
        if line_number == 2683 and "st.warning" in line:
            # Fix indentation
            fixed_lines.append("                st.warning(f\"Scan saved successfully, but prediction results couldn't be stored: {e}\")\n")
            continue
        
        # Skip duplicate import at around line 1644
        if "from visit_comparison import display_visit_comparison" in line and fixed_lines and "from visit_comparison import display_visit_comparison" in "".join(fixed_lines):
            # Skip this line - we already have this import
            continue
            
        # Fix indentation for else block in comparison data section (around line 1650)
        if "else:" in line.strip() and i > 1640 and i < 1660:
            # Find correct indentation from context
            if fixed_lines and any("if len(comparison_data) == 2:" in prev_line for prev_line in fixed_lines[-10:]):
                # Match indentation with the if statement
                indent = "                                "
                fixed_lines.append(f"{indent}else:\n")
                continue
                
        if "st.warning" in line and "Could not retrieve data for both" in line:
            fixed_lines.append("                                    " + line.lstrip())
            continue
            
        if "# Show what we did retrieve" in line:
            fixed_lines.append("                                    " + line.lstrip())
            continue
            
        if "st.write(\"Retrieved data:\"" in line:
            fixed_lines.append("                                    " + line.lstrip())
            continue
        
        # Add line as-is if no special handling needed
        fixed_lines.append(line)
    
    # Write back the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed all indentation errors in {file_path}")

if __name__ == "__main__":
    main() 