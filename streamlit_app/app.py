import streamlit as st
import os
import pathlib
current_dir = pathlib.Path(__file__).parent
# Set page config must be the first streamlit command
st.set_page_config(page_title="Alzheimer Diagnosis System", layout="wide")

import mysql.connector
from admin_view import admin_panel
from doctor_view import doctor_panel
from patient_portal import patient_portal

# Database connection parameters
try:
    from db_config import DB_CONFIG
except ImportError:
    # Fallback configuration if db_config.py is not available
    DB_CONFIG = {
        "host": "localhost",
        "port": 3306,
        "user": "root",
        "password": "root",
        "database": "smart_clinic"
    }

def get_db_connection():
    """Create a direct database connection."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

def login():
    """Handles user authentication with trimmed inputs."""
    # Create a three-column layout with the middle column for login
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Centered logo and title
        st.image("assets/logo.png", width=250)
        st.title("Clinexa")
        st.caption("Beyond Data. Beyond Care.")
        
        # Simple form in a card
        with st.container():
            with st.form("login_form"):
                username = st.text_input("Username").strip()
                password = st.text_input("Password", type="password").strip()
                
                # Add some space
                st.write("")
                
                # Login button
                login_button = st.form_submit_button("Login", use_container_width=True)
                
            # Divider
            st.write("---")
            st.write("or")
            
            # Patient portal button
            if st.button("Patient Portal", use_container_width=True):
                st.session_state.show_patient_portal = True
                st.rerun()
                
            # Help text
            st.caption("Need help? Contact system administrator")
        
    # Process login when button is clicked
    if login_button:
        try:
            conn = get_db_connection()
            if not conn:
                st.error("❌ Failed to connect to the database.")
                return
            
            cursor = conn.cursor()
            
            # Query without hashing the password
            cursor.execute(
                "SELECT id, role, patient_id FROM users WHERE username = %s AND password = %s",
                (username, password)
            )
            result = cursor.fetchone()
            
            if result:
                st.session_state.logged_in = True
                st.session_state.user_id = result[0]
                st.session_state.role = result[1]
                if result[2] is not None:
                    st.session_state.patient_id = result[2]
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()

def main():
    """Main application logic, routes to admin, doctor, or patient views."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    # Check if the patient portal should be shown
    if st.session_state.get("show_patient_portal", False):
        patient_portal()
        # Add a back button to return to main login
        if st.sidebar.button("← Back to Main Login"):
            st.session_state.show_patient_portal = False
            st.rerun()
        return
    
    if not st.session_state.logged_in:
        login()
    else:
        role = st.session_state.get("role", "")
        if role == "admin":
            admin_panel()
        elif role == "doctor":
            doctor_panel()
        elif role == "patient":
            patient_portal()
        else:
            st.error("⚠️ Unknown role. Please contact the system administrator.")
            if st.button("Log out"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()
