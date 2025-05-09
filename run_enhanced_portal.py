import streamlit as st
import sys
import os

# Add current directory to path so we can import the patient_portal module
sys.path.append(os.getcwd())

# Configure page
st.set_page_config(
    page_title="Smart Clinic | Patient Portal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import patient portal
from patient_portal.main import patient_portal

# Add demo data for testing
def setup_demo_data():
    """Setup demo session data for testing"""
    # Add test login credentials
    if "test_mode" not in st.session_state:
        st.session_state.test_mode = True
    
    # Option to auto-login for testing
    auto_login = st.sidebar.checkbox("Auto-login for testing", value=False)
    if auto_login and "logged_in" not in st.session_state:
        # Simulate login with test patient data
        st.session_state.logged_in = True
        st.session_state.user_id = 123
        st.session_state.patient_id = 456
        st.session_state.username = "test_patient"
        st.sidebar.success("Test login activated")
        st.rerun()

# Run portal
def main():
    st.title("Enhanced Patient Portal Demo")
    st.markdown("""
    <div style="padding: 1rem; background-color: #f3f4f6; border-radius: 0.5rem; margin-bottom: 1rem;">
        <h3 style="margin-top: 0;">Demo Mode</h3>
        <p>This is a demonstration of the enhanced patient portal UI.</p>
        <p>Note: This is running with simulated data. In a production environment, it would connect to the full database.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Setup demo data
    setup_demo_data()
    
    # Run the patient portal
    patient_portal()

if __name__ == "__main__":
    main() 