                                            if truncated_scan_type == "axial":
                                                notes = f"Analysis of axial MRI scan shows {prediction_text} with {confidence:.1f}% confidence."
                                            else:
                                                notes = f"Analysis of {truncated_scan_type} MRI scan shows {prediction_text} with {confidence:.1f}% confidence."

                                            # Add medical record
                                            if add_medical_record(patient_id, diagnosis, notes):
                                                st.success("Results added to medical records")
                                else:
                                    # Display error message
                                    error_msg = results.get('error', 'Unknown error occurred') if results else 'Failed to process MRI scan'
                                    st.error(f"Error processing MRI scan: {error_msg}")
                                    
                                    # Still allow saving the scan without analysis
                                    if st.button("Save Scan Without Analysis"): 