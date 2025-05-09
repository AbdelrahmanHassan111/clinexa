import streamlit as st
import pandas as pd
from datetime import datetime
from ..utils.db import get_db_connection, get_patient_info

def sidebar_navigation():
    """
    Create the sidebar navigation for the patient portal.
    Returns the selected page.
    """
    with st.sidebar:
        # Header
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 3rem; color: #3b82f6;">🏥</div>
            <div style="font-weight: 700; font-size: 1.5rem; color: #1e3a8a;">Patient Portal</div>
            <div style="font-size: 0.875rem; color: #6b7280; margin-top: 0.5rem;">Smart Clinic</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Get patient info if available
        if "patient_id" in st.session_state:
            patient_id = st.session_state.patient_id
            patient_info = get_patient_info(patient_id)
            
            if patient_info:
                # Show patient info
                st.markdown(f"""
                <div style="padding: 1rem; background-color: white; border-radius: 0.5rem; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); margin-bottom: 2rem;">
                    <div style="font-weight: 600; font-size: 1.125rem; margin-bottom: 0.5rem; color: #1f2937;">{patient_info.get('full_name', 'Patient')}</div>
                    <div style="font-size: 0.875rem; color: #6b7280; margin-bottom: 0.25rem;">
                        <span style="color: #9ca3af;">ID:</span> {patient_id}
                    </div>
                    <div style="font-size: 0.875rem; color: #6b7280;">
                        <span style="color: #9ca3af;">Member since:</span> {format_date(patient_info.get('created_at', datetime.now()))}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Navigation options
        st.markdown("### Navigation")
        
        # Create buttons with icons for navigation
        nav_options = {
            "Dashboard": "🏠",
            "Appointments": "📅",
            "Medical Records": "📋",
            "Profile": "👤",
            "Sign Out": "🚪"
        }
        
        selected = None
        for page, icon in nav_options.items():
            if st.button(f"{icon} {page}", use_container_width=True, key=f"nav_{page}"):
                selected = page
        
        # Default to Dashboard if nothing is selected
        if selected is None:
            selected = "Dashboard"
            
        # Display info at the bottom of sidebar
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.75rem; color: #9ca3af; text-align: center; margin-top: 2rem;">
            <div>© 2025 Smart Clinic</div>
            <div style="margin-top: 0.5rem;">Version 1.0.0</div>
        </div>
        """, unsafe_allow_html=True)
        
        return selected

def format_date(date):
    """Format date to a readable string."""
    try:
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        return date.strftime("%B %d, %Y")
    except Exception:
        return str(date) 