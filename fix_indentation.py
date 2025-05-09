#!/usr/bin/env python

def main():
    # Define file path
    file_path = "streamlit_app/doctor_view.py"
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix indentation errors
    fixed_lines = []
    for i, line in enumerate(lines):
        # Skip duplicate import at around line 1644
        if i == 1643 and "from visit_comparison import display_visit_comparison" in line:
            # Skip this line
            continue
        # Fix the "else" block at around line 1650
        elif i >= 1649 and line.strip() == "else:":
            fixed_lines.append(line)
        elif i >= 1650 and "st.warning" in line:
            # Add proper indentation (8 spaces more)
            fixed_lines.append("                                        " + line.lstrip())
        elif i >= 1651 and "# Show what" in line:
            # Add proper indentation
            fixed_lines.append("                                        " + line.lstrip())
        elif i >= 1652 and "st.write" in line and "Retrieved data:" in line:
            # Add proper indentation
            fixed_lines.append("                                        " + line.lstrip())
        # Fix the first indentation error (line ~167-168)
        elif "probabilities = clf.predict_proba" in line and not line.startswith("            "):
            # Add proper indentation
            fixed_lines.append("            " + line.lstrip())
        # Fix the second indentation error (line ~1121-1122)
        elif "analysis_id = store_prediction" in line and not line.startswith("                        "):
            # Add proper indentation
            fixed_lines.append("                        " + line.lstrip())
        else:
            fixed_lines.append(line)
    
    # Write back the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed indentation in {file_path}")

if __name__ == "__main__":
    main() 