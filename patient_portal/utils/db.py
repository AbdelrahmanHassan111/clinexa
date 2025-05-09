import streamlit as st
import mysql.connector
import sys
from datetime import datetime

def get_db_connection():
    """Create a database connection using DB_CONFIG."""
    try:
        sys.path.append("..")
        
        try:
            from db_config import DB_CONFIG
        except ImportError:
            # Fallback configuration
            DB_CONFIG = {
                "host": "localhost",
                "port": 3306,
                "user": "root",
                "password": "root",
                "database": "smart_clinic"
            }
        
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

def get_patient_info(patient_id):
    """Get patient information from the database."""
    conn = get_db_connection()
    if not conn:
        return None
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT p.*, 
                   DATE_FORMAT(birth_date, '%Y-%m-%d') as formatted_birth_date,
                   CONCAT(first_name, ' ', last_name) as full_name
            FROM patients p
            WHERE patient_id = %s
        """, (patient_id,))
        
        patient = cursor.fetchone()
        cursor.close()
        conn.close()
        return patient
    except Exception as e:
        st.error(f"Error fetching patient info: {e}")
        cursor.close()
        conn.close()
        return None

def get_patient_appointments(patient_id, limit=10):
    """Get appointments for a patient."""
    conn = get_db_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT a.*, 
                   CONCAT(d.first_name, ' ', d.last_name) as doctor_name,
                   DATE_FORMAT(appointment_date, '%Y-%m-%d') as formatted_date,
                   DATE_FORMAT(appointment_time, '%H:%i') as formatted_time
            FROM appointments a
            LEFT JOIN doctors d ON a.doctor_id = d.doctor_id
            WHERE a.patient_id = %s
            ORDER BY appointment_date DESC, appointment_time DESC
            LIMIT %s
        """, (patient_id, limit))
        
        appointments = cursor.fetchall()
        cursor.close()
        conn.close()
        return appointments
    except Exception as e:
        st.error(f"Error fetching appointments: {e}")
        cursor.close()
        conn.close()
        return []

def get_patient_records(patient_id, limit=20):
    """Get medical records for a patient."""
    conn = get_db_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT mr.*, 
                   DATE_FORMAT(visit_date, '%Y-%m-%d') as formatted_date,
                   CONCAT(d.first_name, ' ', d.last_name) as doctor_name
            FROM medical_records mr
            LEFT JOIN doctors d ON mr.doctor_id = d.doctor_id
            WHERE mr.patient_id = %s
            ORDER BY visit_date DESC
            LIMIT %s
        """, (patient_id, limit))
        
        records = cursor.fetchall()
        cursor.close()
        conn.close()
        return records
    except Exception as e:
        st.error(f"Error fetching medical records: {e}")
        cursor.close()
        conn.close()
        return []

def get_available_doctors():
    """Get available doctors from the database."""
    conn = get_db_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT doctor_id, 
                   CONCAT(first_name, ' ', last_name) as full_name,
                   specialty, 
                   years_experience
            FROM doctors
            WHERE is_active = 1
            ORDER BY years_experience DESC
        """)
        
        doctors = cursor.fetchall()
        cursor.close()
        conn.close()
        return doctors
    except Exception as e:
        st.error(f"Error fetching doctors: {e}")
        cursor.close()
        conn.close()
        return []

def schedule_new_appointment(patient_id, doctor_id, appointment_date, 
                          appointment_time, appointment_type, notes=""):
    """Schedule a new appointment."""
    conn = get_db_connection()
    if not conn:
        return False
    
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO appointments 
            (patient_id, doctor_id, appointment_date, appointment_time, 
             appointment_type, status, notes, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            patient_id, doctor_id, appointment_date, appointment_time,
            appointment_type, "scheduled", notes, datetime.now()
        ))
        
        conn.commit()
        appointment_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return appointment_id
    except Exception as e:
        st.error(f"Error scheduling appointment: {e}")
        cursor.close()
        conn.close()
        return False

def update_appointment(appointment_id, status, notes=None):
    """Update an existing appointment."""
    conn = get_db_connection()
    if not conn:
        return False
    
    cursor = conn.cursor()
    try:
        if notes:
            cursor.execute("""
                UPDATE appointments
                SET status = %s, notes = %s
                WHERE appointment_id = %s
            """, (status, notes, appointment_id))
        else:
            cursor.execute("""
                UPDATE appointments
                SET status = %s
                WHERE appointment_id = %s
            """, (status, appointment_id))
        
        conn.commit()
        success = cursor.rowcount > 0
        cursor.close()
        conn.close()
        return success
    except Exception as e:
        st.error(f"Error updating appointment: {e}")
        cursor.close()
        conn.close()
        return False

def get_patient_cognitive_scores(patient_id, limit=10):
    """Get cognitive test scores for a patient."""
    conn = get_db_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT *,
                   DATE_FORMAT(analyzed_at, '%Y-%m-%d') as formatted_date
            FROM alzheimer_analyses
            WHERE patient_id = %s
            ORDER BY analyzed_at DESC
            LIMIT %s
        """, (patient_id, limit))
        
        scores = cursor.fetchall()
        cursor.close()
        conn.close()
        return scores
    except Exception as e:
        st.error(f"Error fetching cognitive scores: {e}")
        cursor.close()
        conn.close()
        return []

def get_mri_scans(patient_id, limit=10):
    """Get MRI scans for a patient."""
    conn = get_db_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT *,
                   DATE_FORMAT(scan_date, '%Y-%m-%d') as formatted_date
            FROM mri_scans
            WHERE patient_id = %s
            ORDER BY scan_date DESC
            LIMIT %s
        """, (patient_id, limit))
        
        scans = cursor.fetchall()
        cursor.close()
        conn.close()
        return scans
    except Exception as e:
        st.error(f"Error fetching MRI scans: {e}")
        cursor.close()
        conn.close()
        return []

def update_patient_profile(patient_id, email=None, phone=None, address=None):
    """Update patient profile information."""
    conn = get_db_connection()
    if not conn:
        return False
    
    # Build the SQL query dynamically based on provided values
    update_query = "UPDATE patients SET "
    params = []
    
    if email:
        update_query += "email = %s, "
        params.append(email)
    
    if phone:
        update_query += "phone = %s, "
        params.append(phone)
    
    if address:
        update_query += "address = %s, "
        params.append(address)
    
    # Remove trailing comma and space
    update_query = update_query.rstrip(", ")
    
    # Add WHERE clause
    update_query += " WHERE patient_id = %s"
    params.append(patient_id)
    
    cursor = conn.cursor()
    try:
        cursor.execute(update_query, params)
        conn.commit()
        success = cursor.rowcount > 0
        cursor.close()
        conn.close()
        return success
    except Exception as e:
        st.error(f"Error updating profile: {e}")
        cursor.close()
        conn.close()
        return False

def authenticate_patient(username, password):
    """Authenticate a patient login."""
    conn = get_db_connection()
    if not conn:
        return False, None
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT u.id, u.patient_id, u.username, p.first_name, p.last_name
            FROM users u
            JOIN patients p ON u.patient_id = p.patient_id
            WHERE u.username = %s AND u.password = %s AND u.role = 'patient'
        """, (username, password))
        
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user:
            return True, user
        else:
            return False, None
    except Exception as e:
        st.error(f"Error during authentication: {e}")
        cursor.close()
        conn.close()
        return False, None

def register_patient(first_name, last_name, email, phone, birth_date, gender, 
                    address, username, password):
    """Register a new patient."""
    conn = get_db_connection()
    if not conn:
        return False, None
    
    cursor = conn.cursor()
    try:
        # Start a transaction
        conn.start_transaction()
        
        # Insert into patients table
        cursor.execute("""
            INSERT INTO patients 
            (first_name, last_name, birth_date, gender, email, phone, address, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            first_name, last_name, birth_date, gender, 
            email, phone, address, datetime.now()
        ))
        
        # Get the new patient_id
        patient_id = cursor.lastrowid
        
        # Insert into users table
        cursor.execute("""
            INSERT INTO users 
            (username, password, role, patient_id, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            username, password, 'patient', patient_id, datetime.now()
        ))
        
        # Commit the transaction
        conn.commit()
        
        cursor.close()
        conn.close()
        return True, patient_id
    except Exception as e:
        # Rollback in case of error
        conn.rollback()
        st.error(f"Error during registration: {e}")
        cursor.close()
        conn.close()
        return False, None

def get_doctor_appointments(doctor_id, appointment_date):
    """Get doctor's appointments for a specific date."""
    conn = get_db_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT appointment_id, 
                   appointment_date,
                   appointment_time,
                   status,
                   DATE_FORMAT(appointment_time, '%H:%i') as formatted_time
            FROM appointments
            WHERE doctor_id = %s 
              AND DATE(appointment_date) = %s
            ORDER BY appointment_time
        """, (doctor_id, appointment_date))
        
        appointments = cursor.fetchall()
        cursor.close()
        conn.close()
        return appointments
    except Exception as e:
        st.error(f"Error fetching doctor's appointments: {e}")
        cursor.close()
        conn.close()
        return []

def get_patient_appointments_by_date(patient_id, appointment_date):
    """Get a patient's appointments for a specific date."""
    conn = get_db_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT appointment_id, 
                   appointment_date,
                   appointment_time,
                   status,
                   DATE_FORMAT(appointment_time, '%H:%i') as formatted_time
            FROM appointments
            WHERE patient_id = %s 
              AND DATE(appointment_date) = %s
            ORDER BY appointment_time
        """, (patient_id, appointment_date))
        
        appointments = cursor.fetchall()
        cursor.close()
        conn.close()
        return appointments
    except Exception as e:
        st.error(f"Error fetching patient's appointments: {e}")
        cursor.close()
        conn.close()
        return [] 