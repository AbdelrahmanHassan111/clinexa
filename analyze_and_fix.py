#!/usr/bin/env python
import os
import re

def analyze_and_fix():
    file_path = "streamlit_app/doctor_view.py"
    backup_path = "streamlit_app/doctor_view.py.bak2"
    
    # Create a backup of the current file
    if not os.path.exists(backup_path):
        with open(file_path, 'r', encoding='utf-8') as src:
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print(f"Created backup at {backup_path}")
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    
    # Find the tab3 declaration line
    tab3_declaration_line = -1
    for i, line in enumerate(content):
        if "tab1, tab2, tab3 = st.tabs" in line and "Analysis History" in line:
            tab3_declaration_line = i
            break
    
    if tab3_declaration_line == -1:
        print("Could not find tab3 declaration line")
        return
    
    # Find where tab2 implementation ends
    tab2_end_line = -1
    for i in range(tab3_declaration_line + 1, len(content)):
        if "# MRI Analysis Tab" in content[i]:
            tab2_start_line = i
        if "with tab2:" in content[i]:
            tab2_start_line = i
            
    if tab2_start_line == -1:
        print("Could not find tab2 start line")
        return
    
    # Find where tab2 implementation might end
    tab2_end_line = -1
    # Look for lines that might indicate the end of tab2
    potential_tab_end_markers = [
        "# Analysis History tab",
        "with tab3:", 
        "# Display medical records page",
        "def display_medical_records"
    ]
    
    for i in range(tab2_start_line + 1, len(content)):
        for marker in potential_tab_end_markers:
            if marker in content[i]:
                tab2_end_line = i
                break
        if tab2_end_line != -1:
            break
    
    if tab2_end_line == -1:
        print("Could not find tab2 end line")
        return
    
    # Now check if there is a tab3 implementation after tab2 ends
    has_tab3_implementation = False
    for i in range(tab2_end_line, min(tab2_end_line + 20, len(content))):
        if "with tab3:" in content[i]:
            has_tab3_implementation = True
            break
    
    # If there's no tab3 implementation, add it
    if not has_tab3_implementation:
        print("No tab3 implementation found. Adding Analysis History tab...")
        
        # Create the Analysis History tab implementation
        analysis_history_code = """
    # Analysis History tab
    with tab3:
        analyses = get_patient_analyses(patient_id)
        
        if not analyses:
            st.info("No previous analyses found for this patient.")
        else:
            # Add debug info to verify data is coming through
            st.write(f"Found {len(analyses)} previous analyses for this patient.")
            
            # Convert to DataFrame for display and analysis
            df_analyses = pd.DataFrame(analyses)
            
            # Format dates
            df_analyses['analyzed_at'] = pd.to_datetime(df_analyses['analyzed_at'])
            df_analyses['Date'] = df_analyses['analyzed_at'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Prepare for display
            display_df = df_analyses[['analysis_id', 'prediction', 'confidence_score', 'Date']]
            display_df = display_df.rename(columns={
                'analysis_id': 'ID', 
                'prediction': 'Diagnosis', 
                'confidence_score': 'Confidence'
            })
            
            # Format confidence for display
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{float(x):.1%}")
            
            # Show analyses in a table with a sidebar for selection
            st.subheader("Patient Analysis History")
            st.dataframe(display_df.sort_values('Date', ascending=False), use_container_width=True)
            
            # Create a comprehensive comparison section
            st.subheader("Visit Comparison Analysis")
            
            # Allow selection of two analyses to compare
            col1, col2 = st.columns(2)
            
            with col1:
                # Get unique analysis dates with IDs for selection
                analysis_options = {f"{row['Date']} (ID: {row['ID']})": row['ID'] 
                                   for i, row in display_df.iterrows()}
                
                first_analysis = st.selectbox(
                    "Select First Visit", 
                    options=list(analysis_options.keys()),
                    index=min(1, len(analysis_options)-1) if len(analysis_options) > 1 else 0
                )
                first_id = analysis_options[first_analysis]
            
            with col2:
                # Filter out first selection for second dropdown
                remaining_options = {k: v for k, v in analysis_options.items() 
                                   if v != analysis_options[first_analysis]}
                
                if remaining_options:
                    second_analysis = st.selectbox(
                        "Select Second Visit", 
                        options=list(remaining_options.keys()),
                        index=0
                    )
                    second_id = remaining_options[second_analysis]
                    
                    # Enable comparison
                    can_compare = True
                else:
                    st.info("Need at least two analyses to compare.")
                    can_compare = False
            
            # Retrieve full feature data for the selected analyses
            if can_compare:
                if st.button("📊 Compare Selected Visits", type="primary"):
                    try:
                        conn = get_db_connection()
                        if not conn:
                            st.error("Could not connect to database")
                        else:
                            try:
                                cursor = conn.cursor(dictionary=True)
                                # Add debug info
                                st.write(f"Retrieving comparison data for analyses {first_id} and {second_id}")
                                
                                # Fix the SQL query to use proper parameter format - this was likely causing the issue
                                cursor.execute(\"\"\"
                                    SELECT analysis_id, prediction, confidence_score, analyzed_at, input_features
                                    FROM alzheimers_analysis
                                    WHERE analysis_id = %s OR analysis_id = %s
                                \"\"\", (first_id, second_id))
                                
                                comparison_data = cursor.fetchall()
                                cursor.close()
                                conn.close()
                                
                                # Add debug info
                                st.write(f"Retrieved {len(comparison_data)} records for comparison")
                                
                                if len(comparison_data) == 2:
                                    # Use the visit comparison function we created
                                    st.success(f"Comparing visit {first_id} and visit {second_id}")
                                    
                                    try:
                                        from visit_comparison import display_visit_comparison
                                        display_visit_comparison(comparison_data, patient_info, model, get_feature_descriptions())
                                    except Exception as e:
                                        st.error(f"Error in comparison display: {e}")
                                        import traceback
                                        st.error(traceback.format_exc())
                                else:
                                    st.warning(f"Could not retrieve data for both selected visits. Found {len(comparison_data)} records.")
                                    # Show what we did retrieve
                                    st.write("Retrieved data:", comparison_data)
                            except Exception as e:
                                st.error(f"Database error: {e}")
                                import traceback
                                st.error(traceback.format_exc())
                                if conn and hasattr(conn, 'is_connected') and conn.is_connected():
                                    cursor.close()
                                    conn.close()
                    except Exception as e:
                        st.error(f"Error occurred: {e}")
                        import traceback
                        st.error(traceback.format_exc())
                else:
                    st.info("Click the 'Compare Selected Visits' button to see a detailed comparison.")
"""
        
        # Insert the Analysis History tab implementation before the next function 
        # or at the end of the display_alzheimer_analysis function
        content.insert(tab2_end_line, analysis_history_code)
        
        # Write the modified content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(content)
        
        print("Added Analysis History tab implementation")
    else:
        print("tab3 implementation already exists")

if __name__ == "__main__":
    analyze_and_fix() 