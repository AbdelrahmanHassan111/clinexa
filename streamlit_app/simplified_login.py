import streamlit as st
import mysql.connector
import hashlib
import re
from datetime import datetime

# Set page config
st.set_page_config(page_title="Alzheimer Diagnosis System", layout="wide")

# Database connection parameters
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "smart_clinic"
}

# Add custom CSS for improved appearance
st.markdown("""
<style>
    /* Main app styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Center content */
    .center-container {
        max-width: 600px;
        margin: 0 auto;
        padding: 2rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-top: 3rem;
        text-align: center;
    }
    
    /* Header styling */
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .header h1 {
        color: #1e40af;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .header p {
        color: #6b7280;
        font-size: 1.1rem;
    }
    
    /* Logo styling */
    .logo {
        font-size: 5rem;
        margin-bottom: 1rem;
    }
    
    /* Form styling */
    .form-container {
        background-color: #f9fafb;
        padding: 2rem;
        border-radius: 8px;
        text-align: left;
        margin-top: 1.5rem;
    }
    
    /* Button styling */
    .big-button {
        background-color: #2563eb;
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .big-button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }
    
    /* Patient button */
    .patient-button {
        background-color: #059669;
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.25rem;
        transition: all 0.3s ease;
        width: 100%;
        text-align: center;
        margin-top: 2rem;
        cursor: pointer;
    }
    
    .patient-button:hover {
        background-color: #047857;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(5, 150, 105, 0.25);
    }
    
    /* Divider */
    .divider {
        margin: 2rem 0;
        color: #d1d5db;
        display: flex;
        align-items: center;
        text-align: center;
    }
    
    .divider::before, .divider::after {
        content: '';
        flex: 1;
        border-bottom: 1px solid #d1d5db;
    }
    
    .divider::before {
        margin-right: 1rem;
    }
    
    .divider::after {
        margin-left: 1rem;
    }
    
    /* Animated pulse effect */
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(5, 150, 105, 0.4);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(5, 150, 105, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(5, 150, 105, 0);
        }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

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

def authenticate_staff(username, password):
    """Authenticate staff (doctor/admin) users."""
    conn = get_db_connection()
    if not conn:
        return False, None
    
    cursor = conn.cursor(dictionary=True)
    try:
        # Query the users table without role filter
        cursor.execute(
            "SELECT id, username, role, patient_id, doctor_id FROM users WHERE username = %s AND password = %s AND role IN ('doctor', 'admin')",
            (username, hash_password(password))
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
    """Display simplified login page with patient focus."""
    # Center container for main content
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    
    # Logo and title
    st.markdown('<div class="header">', unsafe_allow_html=True)
    st.markdown('<div class="logo">🧠</div>', unsafe_allow_html=True)
    st.markdown('<h1>Alzheimer Diagnosis System</h1>', unsafe_allow_html=True)
    st.markdown('<p>Smart medical diagnostics platform</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main login tabs
    tab1, tab2 = st.tabs(["🩺 Staff Login", "👨‍👩‍👧‍👦 Patient Portal"])
    
    # Staff Login Tab
    with tab1:
        with st.form("staff_login_form"):
            st.markdown('<h3 style="text-align: center; color: #1e40af;">Staff Login</h3>', unsafe_allow_html=True)
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            # Submit button inside form
            submit = st.form_submit_button("Sign In", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    # Authenticate staff
                    success, user_data = authenticate_staff(username, password)
                    
                    if success:
                        # Set session state
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.role = user_data["role"]
                        st.session_state.user_id = user_data["id"]
                        
                        # Set role-specific IDs
                        if user_data["role"] == "doctor":
                            st.session_state.doctor_id = user_data["doctor_id"]
                        
                        st.success(f"Welcome, {username}! Redirecting to {user_data['role']} dashboard...")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
    
    # Patient Portal Tab
    with tab2:
        with st.form("patient_login_form"):
            st.markdown('<h3 style="text-align: center; color: #059669;">Patient Login</h3>', unsafe_allow_html=True)
            patient_email = st.text_input("Email Address")
            patient_password = st.text_input("Password", type="password")
            
            # Submit button inside form
            patient_submit = st.form_submit_button("Access Patient Portal", use_container_width=True)
            
            if patient_submit:
                if not patient_email or not patient_password:
                    st.error("Please enter both email and password")
                else:
                    # Validate email format
                    if not validate_email(patient_email):
                        st.error("Please enter a valid email address")
                    else:
                        # Try to authenticate patient
                        success, user_data = authenticate_patient(patient_email, patient_password)
                        
                        if success:
                            # Set session state
                            st.session_state.logged_in = True
                            st.session_state.username = patient_email
                            st.session_state.role = "patient"
                            st.session_state.user_id = user_data["id"]
                            st.session_state.patient_id = user_data["patient_id"]
                            
                            st.success("Login successful! Redirecting to patient portal...")
                            st.rerun()
                        else:
                            st.error("Invalid email or password")
        
        # Registration link
        st.markdown("""
        <div style="text-align: center; margin-top: 1rem;">
            <p>New patient? <a href="#" onclick="alert('Patient registration will be available soon.')">Register here</a></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Info section with animations
    st.markdown("""
    <div style="margin-top: 2rem; text-align: center;">
        <div style="display: flex; justify-content: space-around; margin: 2rem 0;">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🏥</div>
                <div style="font-weight: 600; color: #1e40af;">Smart Clinic</div>
                <div style="color: #6b7280; font-size: 0.9rem;">Advanced diagnostics</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
                <div style="font-weight: 600; color: #1e40af;">AI Analysis</div>
                <div style="color: #6b7280; font-size: 0.9rem;">Predictive models</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📱</div>
                <div style="font-weight: 600; color: #1e40af;">Easy Access</div>
                <div style="color: #6b7280; font-size: 0.9rem;">Manage your health</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close center-container

def main():
    # Check if user is logged in
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        login_page()
    else:
        # Show the appropriate panel based on user role
        role = st.session_state.role
        
        # This is just a placeholder for now
        st.markdown(f"""
        <div style="text-align: center; padding: 3rem;">
            <h1>Welcome to the {role.capitalize()} Dashboard</h1>
            <p>You are now logged in as {st.session_state.username}</p>
            <p>In a complete application, this would redirect to the appropriate dashboard.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Logout button
        if st.button("Logout", type="primary"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main() 