import streamlit as st
# Set page config must be the first streamlit command
st.set_page_config(page_title="Alzheimer Diagnosis System", layout="wide")

# Import core modules
import mysql.connector
from datetime import datetime
import sys
import os
import re
import hashlib

# Add the parent directory to the Python path so that patient_portal can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import view modules
from admin_view import admin_panel
from doctor_view import doctor_panel

# Try to import the enhanced patient portal package
try:
    from patient_portal.main import patient_portal
except ImportError:
    # If import fails, we'll use the fallback implementation in patient_panel
    patient_portal = None

# Add custom CSS styling for better UI
st.markdown("""
<style>
    /* Main app styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Title styling */
    h1, h2, h3 {
        color: #1e3a8a;
    }
    
    /* Login container */
    .login-container {
        max-width: 800px;
        margin: 2rem auto;
        padding: 2rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Card styling */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #e5e7eb;
    }
    
    /* Button styling */
    .primary-button {
        background-color: #3b82f6;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 500;
        text-align: center;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .primary-button:hover {
        background-color: #2563eb;
    }
    
    /* Patient portal button */
    .patient-button {
        background-color: #0ea5e9;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 500;
        text-align: center;
        cursor: pointer;
        transition: background-color 0.3s;
        font-size: 1.1rem;
        margin-top: 15px;
    }
    
    .patient-button:hover {
        background-color: #0284c7;
    }
</style>
""", unsafe_allow_html=True)

# Database configuration - FIXED CONNECTION SETTINGS
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",  # Changed from empty string to "root"
    "database": "smart_clinic"  # Changed to match the doctor_view.py database name
}

def get_db_connection():
    """Create database connection using DB_CONFIG."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except mysql.connector.Error as err:
        st.error(f"Database connection error: {err}")
        return None

def validate_email(email):
    """Validate email format"""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

def validate_phone(phone):
    """Validate phone number format"""
    # Allow various formats with optional country code
    pattern = r"^(\+\d{1,3}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$"
    return re.match(pattern, phone) is not None

def hash_password(password):
    """Simple password hashing"""
    # In a production environment, use a proper password hashing library
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_patient(email, password):
    """Authenticate patient user based on email and password."""
    conn = get_db_connection()
    if not conn:
        return False, None
    
    cursor = conn.cursor(dictionary=True)
    try:
        # Query users table with patient role filter and using email as username
        cursor.execute(
            "SELECT id, username, role, patient_id FROM users WHERE username = %s AND password = %s AND role = 'patient'",
            (email, hash_password(password))
        )
        
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user:
            return True, user
        else:
            return False, None
    except Exception as e:
        st.error(f"Authentication error: {e}")
        cursor.close()
        conn.close()
        return False, None

def authenticate(username, password, role):
    """Authenticate admin/doctor user based on username, password, and role."""
    conn = get_db_connection()
    if not conn:
        return False, None
    
    cursor = conn.cursor(dictionary=True)
    try:
        # Query the users table with role filter
        cursor.execute(
            "SELECT id, username, role, patient_id, doctor_id FROM users WHERE username = %s AND password = %s AND role = %s",
            (username, hash_password(password), role)
        )
        
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user:
            return True, user
        else:
            return False, None
    except Exception as e:
        st.error(f"Authentication error: {e}")
        cursor.close()
        conn.close()
        return False, None

def login_page():
    """Display simplified login page with clear patient access."""
    # Initialize session state for patient login toggle
    if "show_patient_login" not in st.session_state:
        st.session_state.show_patient_login = False
    
    # Logo and title
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <div style="font-size: 2.5rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0.5rem;">🧠 Alzheimer Diagnosis System</div>
        <div style="font-size: 1rem; color: #6b7280;">Advanced medical diagnostics platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main login container
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # Split into two sections
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Staff login
        st.markdown("""
        <div style="background-color: #f8fafc; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h2 style="font-size: 1.5rem; margin-bottom: 15px; color: #1e3a8a;">Staff Login</h2>
        """, unsafe_allow_html=True)
        
        with st.form("staff_login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            # Role selection for staff
            role = st.selectbox("Login as", ["Doctor", "Admin"], key="role_select")
            role_lowercase = role.lower()
            
            # Center the login button
            col1a, col1b, col1c = st.columns([1, 2, 1])
            with col1b:
                submit = st.form_submit_button("Sign In", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    # Try to authenticate
                    success, user_data = authenticate(username, password, role_lowercase)
                    
                    if success:
                        # Set session state
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.role = role_lowercase
                        st.session_state.user_id = user_data["id"]
                        
                        # Set role-specific IDs
                        if role_lowercase == "doctor":
                            st.session_state.doctor_id = user_data["doctor_id"]
                        
                        st.success(f"Welcome, {username}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Patient portal button
        st.markdown("""
        <div style="background-color: #f0f9ff; padding: 20px; border-radius: 8px; text-align: center; height: 90%; display: flex; flex-direction: column; justify-content: center;">
            <h2 style="font-size: 1.5rem; margin-bottom: 15px; color: #0369a1;">Patient Portal</h2>
            <p style="margin-bottom: 20px; color: #64748b;">Access your health records, appointments, and test results</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Access Patient Portal", type="primary", use_container_width=True):
            st.session_state.show_patient_login = True
            st.rerun()
    
    # Patient login form (shown when button is clicked)
    if st.session_state.show_patient_login:
        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("""
        <h2 style="text-align: center; font-size: 1.5rem; margin: 20px 0; color: #0369a1;">Patient Login</h2>
        """, unsafe_allow_html=True)
        
        # Center the patient login form
        patient_cols = st.columns([1, 2, 1])
        with patient_cols[1]:
            with st.form("patient_login_form"):
                patient_email = st.text_input("Email Address")
                patient_password = st.text_input("Password", type="password")
                
                patient_submit = st.form_submit_button("Sign In", type="primary", use_container_width=True)
                
                if patient_submit:
                    if not patient_email or not patient_password:
                        st.error("Please enter both email and password")
                    else:
                        # Validate email format
                        if not validate_email(patient_email):
                            st.error("Please enter a valid email address")
                        else:
                            # Try to authenticate
                            success, user_data = authenticate_patient(patient_email, patient_password)
                            
                            if success:
                                # Set session state
                                st.session_state.logged_in = True
                                st.session_state.username = patient_email
                                st.session_state.role = "patient"
                                st.session_state.user_id = user_data["id"]
                                st.session_state.patient_id = user_data["patient_id"]
                                
                                st.success("Login successful!")
                                st.rerun()
                            else:
                                st.error("Invalid email or password")
            
            # Registration link
            st.markdown("""
            <div style="text-align: center; margin-top: 10px;">
                <p>New patient? <a href="#" onclick="alert('Registration form will open')">Register here</a></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Option to hide patient login
            if st.button("Cancel"):
                st.session_state.show_patient_login = False
                st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

def patient_panel():
    """Launch the enhanced patient portal with simplified UI."""
    if patient_portal:
        # Use the imported patient portal if available
        patient_portal()
    else:
        # Fallback implementation if module import failed
        st.title("🏥 Patient Portal")
        
        # Check if user is logged in as patient
        if "logged_in" not in st.session_state or not st.session_state.logged_in or st.session_state.role != "patient":
            st.warning("Please log in as a patient to access the patient portal.")
            return
        
        # Add custom CSS for a cleaner look
        st.markdown("""
        <style>
            .patient-header {
                background: linear-gradient(135deg, #0ea5e9, #0369a1);
                color: white;
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1.5rem;
            }
            .metric-card {
                background-color: white;
                border-radius: 8px;
                padding: 16px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                margin-bottom: 16px;
                border: 1px solid #f0f0f0;
            }
            .metric-value {
                font-size: 1.8rem;
                font-weight: 600;
                color: #0369a1;
            }
            .metric-label {
                color: #64748b;
                font-size: 0.9rem;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Display patient header with info
        st.markdown(f"""
        <div class="patient-header">
            <h2>Welcome, {st.session_state.username}</h2>
            <p>Patient ID: {st.session_state.patient_id}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Basic navigation
        nav_options = ["Dashboard", "Appointments", "Medical Records", "Profile", "Sign Out"]
        selected_option = st.sidebar.radio("Navigation", nav_options)
        
        if selected_option == "Dashboard":
            st.header("Patient Dashboard")
            
            # Display metrics in a grid
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Upcoming Appointments</div>
                    <div class="metric-value">2</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Cognitive Assessment</div>
                    <div class="metric-value">28/30</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">MRI Scans</div>
                    <div class="metric-value">3</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Recent activity
            st.subheader("Recent Activity")
            st.markdown("""
            - **MRI Scan Analysis** - Completed on May 8, 2023
            - **Cognitive Assessment** - Scheduled for May 15, 2023
            - **Doctor Appointment** - Dr. Johnson on June 3, 2023
            """)
            
        elif selected_option == "Appointments":
            st.header("Appointments")
            
            # Tabs for viewing/scheduling
            appt_tab1, appt_tab2 = st.tabs(["My Appointments", "Schedule New"])
            
            with appt_tab1:
                st.subheader("Upcoming Appointments")
                st.markdown("""
                <div style="background-color: #f0f9ff; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
                    <div style="font-weight: 600; font-size: 1.1rem;">Dr. Smith - Regular Checkup</div>
                    <div style="color: #64748b;">May 15, 2023 - 10:00 AM</div>
                    <div style="margin-top: 8px; display: inline-block; background-color: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">Scheduled</div>
                </div>
                
                <div style="background-color: #f0f9ff; padding: 16px; border-radius: 8px;">
                    <div style="font-weight: 600; font-size: 1.1rem;">Dr. Johnson - Cognitive Assessment</div>
                    <div style="color: #64748b;">June 3, 2023 - 2:30 PM</div>
                    <div style="margin-top: 8px; display: inline-block; background-color: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">Scheduled</div>
                </div>
                """, unsafe_allow_html=True)
            
            with appt_tab2:
                st.subheader("Schedule New Appointment")
                with st.form("schedule_appointment"):
                    st.selectbox("Doctor", ["Dr. Smith (Neurology)", "Dr. Johnson (Geriatrics)"])
                    st.date_input("Date")
                    st.selectbox("Time", ["9:00 AM", "10:00 AM", "11:00 AM", "1:00 PM", "2:00 PM", "3:00 PM"])
                    st.text_area("Reason for Visit")
                    st.form_submit_button("Schedule Appointment")
            
        elif selected_option == "Medical Records":
            st.header("Medical Records")
            
            # Tabs for different types of records
            rec_tab1, rec_tab2, rec_tab3 = st.tabs(["Clinical Records", "Brain Scans", "Cognitive Tests"])
            
            with rec_tab1:
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
            
            with rec_tab2:
                st.image("https://www.mayoclinic.org/-/media/kcms/gbs/patient-consumer/images/2013/11/15/17/39/ds00853_-ds00862_-ds01058_im02220_ms7_mri_brainkl_jpg.jpg", 
                        caption="MRI Scan - March 5, 2023")
            
            with rec_tab3:
                st.markdown("""
                ### Mini-Mental State Examination (MMSE)
                **Score**: 28/30  
                **Date**: April 10, 2023  
                **Interpretation**: Normal to Mild Cognitive Impairment
                
                ### Clinical Dementia Rating (CDR)
                **Score**: 0.5  
                **Date**: April 10, 2023  
                **Interpretation**: Questionable Dementia
                """)
                
        elif selected_option == "Profile":
            st.header("Patient Profile")
            
            # Tabs for profile sections
            profile_tab1, profile_tab2 = st.tabs(["Personal Information", "Account Settings"])
            
            with profile_tab1:
                with st.form("update_profile_form"):
                    st.text_input("Full Name", value="John Doe")
                    st.text_input("Email Address", value="john.doe@example.com")
                    st.text_input("Phone Number", value="555-123-4567")
                    st.text_area("Address", value="123 Main St, Anytown, US 12345")
                    st.form_submit_button("Update Profile")
            
            with profile_tab2:
                st.subheader("Change Password")
                with st.form("change_password_form"):
                    st.text_input("Current Password", type="password")
                    st.text_input("New Password", type="password")
                    st.text_input("Confirm New Password", type="password")
                    st.form_submit_button("Change Password")
                
        elif selected_option == "Sign Out":
            st.session_state.clear()
            st.rerun()

def main():
    """Main application entry point."""
    # Check if user is logged in
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        login_page()
    else:
        # Show the appropriate panel based on user role
        role = st.session_state.role
        
        if role == "doctor":
            doctor_panel()
        elif role == "patient":
            patient_panel()
        elif role == "admin":
            admin_panel()
        else:
            st.error("Unknown role. Please log in again.")
            st.session_state.clear()
            st.rerun()
        
        # Logout button in sidebar
        with st.sidebar:
            if st.button("Logout", key="logout"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main() 