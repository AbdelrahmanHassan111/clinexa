import streamlit as st
import os
import sys

# Ensure patient_portal is in the path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import from the regular patient_portal package
try:
    from patient_portal.main import patient_portal as original_patient_portal
    patient_portal = original_patient_portal
    print("Successfully imported patient_portal.main")
except ImportError as e:
    print(f"Could not import original patient_portal.main: {e}")
    # Fallback simplified implementation if the import fails
    def patient_portal():
        """Simplified fallback patient portal implementation"""
        st.title("🏥 Patient Portal (Simplified Version)")
        
        # Check if user is logged in
        if "logged_in" not in st.session_state or not st.session_state.logged_in or st.session_state.role != "patient":
            st.warning("Please log in as a patient to access the patient portal.")
            return
        
        # Basic navigation
        nav_options = ["Dashboard", "Appointments", "Medical Records", "Profile", "Sign Out"]
        selected_option = st.sidebar.radio("Navigation", nav_options)
        
        if selected_option == "Dashboard":
            st.header("Patient Dashboard")
            st.info("Welcome to your patient dashboard. This is a simplified version of the patient portal.")
            
            # Display some basic metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Upcoming Appointments", "2")
            with col2:
                st.metric("Medical Records", "5")
            with col3:
                st.metric("Cognitive Score", "28/30")
                
        elif selected_option == "Appointments":
            st.header("Appointments")
            st.info("View and manage your appointments here.")
            
            # Sample appointments
            st.subheader("Upcoming Appointments")
            st.markdown("""
            - **May 15, 2023 - 10:00 AM** - Dr. Smith - Regular Checkup
            - **June 3, 2023 - 2:30 PM** - Dr. Johnson - Cognitive Assessment
            """)
            
        elif selected_option == "Medical Records":
            st.header("Medical Records")
            st.info("View your medical records here.")
            
            # Sample medical records
            st.subheader("Recent Records")
            with st.expander("April 10, 2023 - Cognitive Assessment"):
                st.markdown("""
                **Diagnosis**: Mild Cognitive Impairment  
                **Notes**: Patient shows minor memory issues but maintains daily function.
                """)
                
            with st.expander("March 5, 2023 - MRI Scan"):
                st.markdown("""
                **Diagnosis**: Normal Scan  
                **Notes**: No significant abnormalities detected in brain structure.
                """)
                
        elif selected_option == "Profile":
            st.header("Patient Profile")
            st.info("View and edit your profile information here.")
            
            # Sample profile form
            with st.form("profile_form"):
                st.text_input("Name", value="John Doe", disabled=True)
                st.text_input("Email", value="john.doe@example.com")
                st.text_input("Phone", value="555-123-4567")
                st.form_submit_button("Update Profile")
                
        elif selected_option == "Sign Out":
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Smart Clinic | Patient Portal",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    patient_portal() 