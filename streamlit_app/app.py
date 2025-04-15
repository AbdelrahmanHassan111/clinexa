import streamlit as st
import mysql.connector
from admin_view import admin_panel
from doctor_view import doctor_panel

# Database connection parameters
DB_CONFIG = {
    "host": "localhost",
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
    st.set_page_config(page_title="Alzheimer Diagnosis System", layout="centered")
    st.title("🧠 Alzheimer Diagnosis System")
    st.subheader("Secure Login")
    
    username = st.text_input("Username").strip()  # Remove leading/trailing spaces
    password = st.text_input("Password", type="password").strip()  # Remove spaces
    
    if st.button("Login"):
        try:
            conn = get_db_connection()
            if not conn:
                st.error("❌ Failed to connect to the database.")
                return
            
            cursor = conn.cursor()
            
            # Query without hashing the password
            cursor.execute(
                "SELECT id, role FROM users WHERE username = %s AND password = %s",
                (username, password)
            )
            result = cursor.fetchone()
            
            if result:
                st.session_state.logged_in = True
                st.session_state.user_id = result[0]
                st.session_state.role = result[1]
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
    """Main application logic, routes to admin or doctor views."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        login()
    else:
        role = st.session_state.get("role", "")
        if role == "admin":
            admin_panel()
        elif role == "doctor":
            doctor_panel()
        else:
            st.error("⚠️ Unknown role. Please contact the system administrator.")
            if st.button("Log out"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()