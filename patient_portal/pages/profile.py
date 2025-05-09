import streamlit as st
from datetime import datetime
import pandas as pd
from ..utils.db import get_patient_info, update_patient_profile

def profile_page(patient_id):
    """Display and allow editing of patient profile information."""
    
    # Page header
    st.markdown("""
    <div class="dashboard-header">
        <h1>My Profile</h1>
        <p>Manage your personal information and account settings</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get patient information
    patient_info = get_patient_info(patient_id)
    
    if not patient_info:
        st.error("Could not retrieve patient information. Please try again later.")
        return
    
    # Create tabs for different sections of profile
    tab1, tab2, tab3 = st.tabs(["Personal Information", "Account Settings", "Health Preferences"])
    
    # Personal Information Tab
    with tab1:
        personal_info_tab(patient_info, patient_id)
    
    # Account Settings Tab
    with tab2:
        account_settings_tab(patient_info)
    
    # Health Preferences Tab
    with tab3:
        health_preferences_tab(patient_info, patient_id)

def personal_info_tab(patient_info, patient_id):
    """Display and edit personal information."""
    
    st.markdown("<div class='section-title'>Personal Details</div>", unsafe_allow_html=True)
    
    # Create profile card
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Profile image/avatar
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <div style="width: 120px; height: 120px; border-radius: 50%; background-color: #e5e7eb; margin: 0 auto; display: flex; align-items: center; justify-content: center; overflow: hidden;">
                <div style="font-size: 3rem; color: #9ca3af;">👤</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display patient ID card
        st.markdown(f"""
        <div style="text-align: center; background-color: #eff6ff; padding: 1rem; border-radius: 0.5rem; border: 1px solid #3b82f6;">
            <div style="font-size: 0.75rem; color: #3b82f6; text-transform: uppercase; font-weight: 600; margin-bottom: 0.5rem;">Patient ID</div>
            <div style="font-size: 1.25rem; font-weight: 600; color: #1e3a8a;">{patient_info.get('patient_id', 'Unknown')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Display personal information
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Basic Information</div>
            
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 0.5rem 0; color: #6b7280; width: 30%;">Name</td>
                    <td style="padding: 0.5rem 0; font-weight: 500;">{patient_info.get('full_name', 'Unknown')}</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0; color: #6b7280;">Gender</td>
                    <td style="padding: 0.5rem 0; font-weight: 500;">{patient_info.get('gender', 'Unknown')}</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0; color: #6b7280;">Date of Birth</td>
                    <td style="padding: 0.5rem 0; font-weight: 500;">{format_date(patient_info.get('birth_date', 'Unknown'))}</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0; color: #6b7280;">Age</td>
                    <td style="padding: 0.5rem 0; font-weight: 500;">{calculate_age(patient_info.get('birth_date', None))} years</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    # Contact information section with edit
    st.markdown("<div class='section-title'>Contact Information</div>", unsafe_allow_html=True)
    
    # Display current contact info
    current_email = patient_info.get('email', '')
    current_phone = patient_info.get('phone', '')
    current_address = patient_info.get('address', '')
    
    # Create edit form
    with st.form("edit_contact_form"):
        st.markdown("""
        <div style="font-weight: 500; margin-bottom: 1rem;">Edit your contact information below:</div>
        """, unsafe_allow_html=True)
        
        email = st.text_input("Email Address", value=current_email)
        phone = st.text_input("Phone Number", value=current_phone)
        address = st.text_area("Address", value=current_address, height=100)
        
        col1, col2 = st.columns(2)
        
        with col1:
            submit = st.form_submit_button("Update Contact Info", use_container_width=True)
        
        if submit:
            # Validate form
            if not email or not validate_email(email):
                st.error("Please enter a valid email address.")
            elif not phone:
                st.error("Please enter a phone number.")
            else:
                # Update profile
                success = update_patient_profile(patient_id, email, phone, address)
                
                if success:
                    st.success("Contact information updated successfully!")
                    # Update displayed info
                    st.rerun()
                else:
                    st.error("Failed to update contact information. Please try again.")

def account_settings_tab(patient_info):
    """Display account settings with password change option."""
    
    st.markdown("<div class='section-title'>Account Information</div>", unsafe_allow_html=True)
    
    # Account details card
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Account Details</div>
        
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 0.5rem 0; color: #6b7280; width: 30%;">Username</td>
                <td style="padding: 0.5rem 0; font-weight: 500;">{st.session_state.get('username', 'Unknown')}</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem 0; color: #6b7280;">Account Type</td>
                <td style="padding: 0.5rem 0; font-weight: 500;">Patient</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem 0; color: #6b7280;">Member Since</td>
                <td style="padding: 0.5rem 0; font-weight: 500;">{format_date(patient_info.get('created_at', 'Unknown'))}</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem 0; color: #6b7280;">Last Login</td>
                <td style="padding: 0.5rem 0; font-weight: 500;">{format_date(datetime.now())}</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # Change password section
    st.markdown("<div class='section-title'>Change Password</div>", unsafe_allow_html=True)
    
    with st.form("change_password_form"):
        st.markdown("""
        <div style="font-weight: 500; margin-bottom: 1rem;">Update your password:</div>
        """, unsafe_allow_html=True)
        
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        submit = st.form_submit_button("Change Password", use_container_width=True)
        
        if submit:
            # Validate form
            if not current_password:
                st.error("Please enter your current password.")
            elif not new_password:
                st.error("Please enter a new password.")
            elif new_password != confirm_password:
                st.error("New passwords do not match.")
            elif len(new_password) < 6:
                st.error("New password must be at least 6 characters long.")
            else:
                # In a real app, this would call an API to change the password
                # For this demo, just show success
                st.success("Password changed successfully! (Demo only - not actually changed)")
    
    # Account deletion (just for UI demo, won't actually delete)
    st.markdown("<div class='section-title'>Danger Zone</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 1rem; background-color: #fee2e2; border-radius: 0.5rem; border: 1px solid #ef4444;">
        <div style="font-weight: 600; margin-bottom: 1rem; color: #b91c1c;">Delete Account</div>
        <div style="margin-bottom: 1rem; font-size: 0.875rem; color: #7f1d1d;">
            Permanently delete your account and all associated data. This action cannot be undone.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Delete Account", type="primary", use_container_width=True):
        # Show confirmation dialog
        st.warning("Are you sure you want to delete your account? This action cannot be undone.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, Delete My Account", key="confirm_delete", use_container_width=True):
                # In a real app, this would delete the account
                # For this demo, just show info message
                st.info("This is a demo - account deletion has been simulated. In a real application, your account would be deleted.")
        with col2:
            if st.button("No, Keep My Account", key="cancel_delete", use_container_width=True):
                st.rerun()

def health_preferences_tab(patient_info, patient_id):
    """Display health preferences and notification settings."""
    
    st.markdown("<div class='section-title'>Health Information</div>", unsafe_allow_html=True)
    
    # Display health card with key metrics if available
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">Health Metrics</div>
            
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 0.5rem 0; color: #6b7280; width: 50%;">Height</td>
                    <td style="padding: 0.5rem 0; font-weight: 500;">Not recorded</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0; color: #6b7280;">Weight</td>
                    <td style="padding: 0.5rem 0; font-weight: 500;">Not recorded</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0; color: #6b7280;">Blood Type</td>
                    <td style="padding: 0.5rem 0; font-weight: 500;">Not recorded</td>
                </tr>
            </table>
            
            <div style="margin-top: 1rem; text-align: center;">
                <button style="background-color: #3b82f6; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.25rem; cursor: pointer; font-size: 0.875rem;">
                    Update Health Information
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Medical Conditions</div>
            
            <div style="color: #6b7280; margin-bottom: 1rem;">
                No medical conditions recorded. Please update your health profile with any relevant conditions.
            </div>
            
            <div style="margin-top: 1rem; text-align: center;">
                <button style="background-color: #3b82f6; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.25rem; cursor: pointer; font-size: 0.875rem;">
                    Add Medical Conditions
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Notification preferences
    st.markdown("<div class='section-title'>Notification Settings</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <div class="card-title">Communication Preferences</div>
        
        <div style="margin-bottom: 1rem;">
            Choose how you want to receive notifications and reminders.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Notification preferences
    notify_appointment = st.checkbox("Appointment Reminders", value=True, 
                                    help="Receive reminders before scheduled appointments")
    notify_results = st.checkbox("Test Results", value=True, 
                                help="Get notified when new test results are available")
    notify_messages = st.checkbox("Doctor Messages", value=True, 
                                help="Receive notifications for messages from your doctor")
    notify_educational = st.checkbox("Educational Content", value=False, 
                                    help="Receive educational information about your condition")
    
    # Notification methods
    st.markdown("<div style='margin-top: 1rem; font-weight: 500;'>Notification Methods</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        notify_email = st.checkbox("Email", value=True)
        notify_sms = st.checkbox("SMS/Text Message", value=True)
    
    with col2:
        notify_app = st.checkbox("Mobile App Push", value=False)
        notify_call = st.checkbox("Phone Call", value=False)
    
    # Save preferences button
    if st.button("Save Preferences", use_container_width=True, type="primary"):
        # In a real app, this would save to the database
        # For this demo, just show success message
        st.success("Notification preferences saved successfully!")

def format_date(date_value):
    """Format date to a readable string."""
    try:
        if isinstance(date_value, str):
            date_value = datetime.strptime(date_value, "%Y-%m-%d")
        
        if isinstance(date_value, datetime):
            return date_value.strftime("%B %d, %Y")
    except Exception:
        pass
    
    return str(date_value)

def calculate_age(birth_date):
    """Calculate age from birth date."""
    if not birth_date:
        return "Unknown"
    
    try:
        if isinstance(birth_date, str):
            birth_date = datetime.strptime(birth_date, "%Y-%m-%d")
        
        today = datetime.now()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age
    except:
        return "Unknown"

def validate_email(email):
    """Simple email validation."""
    return "@" in email and "." in email.split("@")[1] 