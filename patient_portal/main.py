import streamlit as st
import os
from .pages.auth import login_page, register_page
from .pages.dashboard import patient_dashboard
from .pages.appointments import schedule_appointment_page
from .pages.medical_records import medical_records_page
from .pages.profile import profile_page
from .components.navigation import sidebar_navigation

def patient_portal():
    """Main entry point for the enhanced patient portal"""
    
    # Load custom CSS
    css_path = "patient_portal/static/style.css"
    if os.path.exists(css_path):
        try:
            with open(css_path, "r") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading CSS: {e}")
    else:
        # Fallback minimal styling if CSS file doesn't exist
        st.markdown("""
        <style>
            .dashboard-header { 
                background: linear-gradient(135deg, #3b82f6, #1e3a8a);
                color: white;
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1.5rem;
            }
            .dashboard-header h1 { color: white !important; }
            .section-title { 
                font-size: 1.25rem;
                font-weight: 600;
                margin: 1.5rem 0 1rem 0;
                color: #1e3a8a;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid #e5e7eb;
            }
        </style>
        """, unsafe_allow_html=True)
    
    # Add custom font and icon imports
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """, unsafe_allow_html=True)
    
    # Check authentication status
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        # Authentication pages
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            login_page()
        
        with tab2:
            register_page()
    else:
        # Patient is logged in - show dashboard
        
        # Get sidebar navigation
        selected_page = sidebar_navigation()
        
        # Display selected page
        if selected_page == "Dashboard":
            patient_dashboard(st.session_state.patient_id)
        
        elif selected_page == "Appointments":
            schedule_appointment_page(st.session_state.patient_id)
        
        elif selected_page == "Medical Records":
            medical_records_page(st.session_state.patient_id)
        
        elif selected_page == "Profile":
            profile_page(st.session_state.patient_id)
        
        elif selected_page == "Sign Out":
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