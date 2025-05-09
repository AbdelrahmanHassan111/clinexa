import streamlit as st
import re
from datetime import datetime, date
from ..utils.db import authenticate_patient, register_patient

def login_page():
    """Patient login page with enhanced UI."""
    
    st.markdown("""
        <div class="auth-container">
            <div class="auth-header">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🧠</div>
                <h1>Welcome Back</h1>
                <p>Sign in to access your Smart Clinic account</p>
            </div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            remember = st.checkbox("Remember me")
        
        submitted = st.form_submit_button("Sign In", use_container_width=True)
        
        if submitted:
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                # Call authentication function
                success, user_data = authenticate_patient(username, password)
                
                if success:
                    # Set session state
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_data['id']
                    st.session_state.patient_id = user_data['patient_id']
                    st.session_state.username = user_data['username']
                    
                    st.success(f"Welcome back, {user_data['first_name']}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    st.markdown("""
        </div>
    """, unsafe_allow_html=True)

def register_page():
    """Patient registration page with enhanced UI and validation."""
    
    st.markdown("""
        <div class="auth-container">
            <div class="auth-header">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📝</div>
                <h1>Create Account</h1>
                <p>Join Smart Clinic for personalized healthcare</p>
            </div>
    """, unsafe_allow_html=True)
    
    with st.form("registration_form"):
        # Personal Information Section
        st.subheader("Personal Information")
        
        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input("First Name", placeholder="Enter your first name")
        with col2:
            last_name = st.text_input("Last Name", placeholder="Enter your last name")
        
        col1, col2 = st.columns(2)
        with col1:
            birth_date = st.date_input("Date of Birth", min_value=date(1900, 1, 1), max_value=date.today())
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        # Contact Information Section
        st.subheader("Contact Information")
        
        email = st.text_input("Email", placeholder="Enter your email address")
        phone = st.text_input("Phone Number", placeholder="Enter your phone number")
        address = st.text_area("Address", placeholder="Enter your address", height=100)
        
        # Account Information Section
        st.subheader("Account Information")
        
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username", placeholder="Choose a username")
        with col2:
            password = st.text_input("Password", type="password", placeholder="Create a password")
        
        # Confirm password
        password_confirm = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
        
        # Terms and Conditions
        terms = st.checkbox("I agree to the Terms and Conditions and Privacy Policy")
        
        submitted = st.form_submit_button("Create Account", use_container_width=True)
        
        if submitted:
            # Validate form
            errors = []
            
            if not first_name or not last_name:
                errors.append("Please enter your first and last name")
            
            if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                errors.append("Please enter a valid email address")
            
            if not phone or not re.match(r"^\d{10,12}$", re.sub(r"[+\-\s()]", "", phone)):
                errors.append("Please enter a valid phone number")
            
            if not username or len(username) < 4:
                errors.append("Username must be at least 4 characters long")
            
            if not password or len(password) < 6:
                errors.append("Password must be at least 6 characters long")
            
            if password != password_confirm:
                errors.append("Passwords do not match")
            
            if not terms:
                errors.append("You must agree to the Terms and Conditions")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Register patient
                success, patient_id = register_patient(
                    first_name, 
                    last_name, 
                    email, 
                    phone, 
                    birth_date,
                    gender,
                    address,
                    username,
                    password
                )
                
                if success:
                    st.success("Registration successful! You can now log in.")
                    # Clear form
                    st.session_state.clear()
                else:
                    st.error("Registration failed. Please try again later or contact support.")
    
    st.markdown("""
        </div>
    """, unsafe_allow_html=True) 