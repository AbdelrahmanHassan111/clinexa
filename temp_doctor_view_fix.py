                    if st.button("Process Scan", type="primary"):
                        with st.spinner("Processing MRI scan..."):
                            # Run Hugging Face model inference
                            st.info("Running Hugging Face model analysis...")
                            from mri_models import process_mri_with_huggingface
                            
                            # Truncate scan_type to 45 characters to prevent database error
                            truncated_scan_type = scan_type[:45] if scan_type and len(scan_type) > 45 else scan_type
                            
                            results = process_mri_with_huggingface(file_path)
                            
                            # Code here for handling results...
                            
                            # Add additional debug information button
                            with st.expander("Debug Information"):
                                st.write("File path:", file_path)
                                st.write("Scan type:", truncated_scan_type)
                                st.write("Original scan type:", scan_type)
                                st.write("Results:", results)
                    else:
                        # Just save the scan without analysis
                        save_result = save_mri_scan(
                            st.session_state.selected_patient, 
                            file_path, 
                            truncated_scan_type, 
                            notes=scan_notes
                        )
                        
                        if save_result:
                            st.success(f"MRI scan saved to patient record with ID #{save_result}")
                        else:
                            st.error("Failed to save scan to database") 