import streamlit as st
import pandas as pd
import mysql.connector
from datetime import datetime, timedelta, time
import re
import hashlib
import plotly.express as px
import plotly.graph_objects as go

# Set page config only when run directly, not when imported
if __name__ == "__main__":
    st.set_page_config(page_title="Smart Clinic Patient Portal", layout="wide", page_icon="🏥")

# Database connection parameters
try:
    from db_config import DB_CONFIG
except ImportError:
    # Fallback configuration if db_config.py is not available
    DB_CONFIG = {
    "host": st.secrets["host"],
    "port": st.secrets["port"],
    "user": st.secrets["username"],
    "password": st.secrets["password"],
    "database": st.secrets["database"]
}

# Add custom CSS for better UI
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Card styling */
    .css-1r6slb0, .css-1lcbmhc {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e3a8a;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        height: 100%;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1e3a8a;
        margin-top: 10px;
    }
    
    .metric-label {
        font-size: 14px;
        color: #6b7280;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-163ttbj {
        background-color: #1e3a8a;
    }
    
    .sidebar .sidebar-content {
        background-color: #1e3a8a;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #1e3a8a;
        color: white;
        border: none;
    }
    
    /* Form input styling */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 5px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f3f4f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-radius: 4px 4px 0px 0px;
        border-right: none;
        border-left: none;
        border-top: 3px solid #1e3a8a;
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)

def get_db_connection():
    """Create a direct database connection."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        st.error(f"Database connection error: {e}")
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

def register_patient():
    """Handle patient registration"""
    st.write("## Create Your Patient Account")
    st.write("Please fill in all required information to register.")
    
    with st.form("registration_form"):
        # Personal Information
        st.write("### Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input("First Name*")
            birth_date = st.date_input("Date of Birth*", max_value=datetime.now().date())
            email = st.text_input("Email Address*")
            password = st.text_input("Create Password*", type="password")
        
        with col2:
            last_name = st.text_input("Last Name*")
            gender = st.selectbox("Gender*", ["Male", "Female", "Other", "Prefer not to say"])
            phone = st.text_input("Phone Number*")
            confirm_password = st.text_input("Confirm Password*", type="password")
        
        # Address
        st.write("### Address")
        address = st.text_area("Home Address*")
        
        # Medical Information
        st.write("### Medical Information")
        col3, col4 = st.columns(2)
        with col3:
            emergency_contact = st.text_input("Emergency Contact Name")
            allergies = st.text_area("Allergies (if any)")
        
        with col4:
            emergency_phone = st.text_input("Emergency Contact Phone")
            medical_conditions = st.text_area("Pre-existing Medical Conditions")
        
        # Terms and conditions
        agree = st.checkbox("I agree to the terms and conditions and privacy policy*")
        
        # Submit button without a key parameter
        submit = st.form_submit_button("Register Account")
        
        if submit:
            # Validate required fields
            if not (first_name and last_name and birth_date and email and phone and address and password):
                st.error("Please fill in all required fields marked with *")
                return False
            
            # Validate email format
            if not validate_email(email):
                st.error("Please enter a valid email address")
                return False
            
            # Validate phone format
            if not validate_phone(phone):
                st.error("Please enter a valid phone number")
                return False
            
            # Validate password match
            if password != confirm_password:
                st.error("Passwords do not match")
                return False
            
            # Validate terms agreement
            if not agree:
                st.error("You must agree to the terms and conditions to register")
                return False
            
            # Connect to database
            conn = get_db_connection()
            if not conn:
                st.error("Database connection failed")
                return False
            
            cursor = conn.cursor()
            
            try:
                # Check if email already exists
                cursor.execute("SELECT COUNT(*) FROM patients WHERE email = %s", (email,))
                if cursor.fetchone()[0] > 0:
                    st.error("An account with this email already exists")
                    cursor.close()
                    conn.close()
                    return False
                
                # Format full name
                full_name = f"{first_name} {last_name}"
                
                # Register the patient in patients table
                cursor.execute("""
                    INSERT INTO patients 
                    (full_name, birth_date, gender, contact_info, email, address, created_at, 
                     emergency_contact, emergency_phone, allergies, medical_conditions)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    full_name, birth_date, gender, phone, email, address, datetime.now(),
                    emergency_contact, emergency_phone, allergies, medical_conditions
                ))
                
                # Get the new patient ID
                patient_id = cursor.lastrowid
                
                # Create user account for the patient
                cursor.execute("""
                    INSERT INTO users
                    (username, password, role, patient_id)
                    VALUES (%s, %s, %s, %s)
                """, (
                    email, hash_password(password), "patient", patient_id
                ))
                
                conn.commit()
                st.success("✅ Registration successful! You can now log in with your email and password.")
                
                # Store in session state for immediate login
                st.session_state.temp_registered = True
                st.session_state.temp_email = email
                
                cursor.close()
                conn.close()
                return True
                
            except Exception as e:
                st.error(f"Error during registration: {e}")
                cursor.close()
                conn.close()
                return False

def get_available_doctors():
    """Get list of available doctors"""
    conn = get_db_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT doctor_id, full_name, specialization
            FROM doctors
            ORDER BY full_name
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

def get_doctor_schedule(doctor_id, start_date, end_date):
    """Get a doctor's schedule for the specified date range"""
    conn = get_db_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT appointment_date
            FROM appointments
            WHERE doctor_id = %s AND appointment_date BETWEEN %s AND %s
        """, (doctor_id, start_date, end_date))
        
        appointments = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Extract just the appointment times
        scheduled_times = [appt['appointment_date'] for appt in appointments]
        return scheduled_times
    except Exception as e:
        st.error(f"Error fetching doctor schedule: {e}")
        cursor.close()
        conn.close()
        return []

def generate_available_slots(doctor_id, selected_date, patient_id=None):
    """
    Generate available time slots for the selected date
    
    Args:
        doctor_id: The ID of the doctor
        selected_date: The date to check for availability
        patient_id: The patient ID to check for existing appointments across all doctors
        
    Returns:
        List of available datetime slots
    """
    # Define clinic hours (9 AM to 5 PM with 30-minute slots)
    start_hour = 9
    end_hour = 17
    slot_duration = 30  # minutes
    
    # Get doctor's existing appointments for the day
    start_datetime = datetime.combine(selected_date, time(0, 0))
    end_datetime = datetime.combine(selected_date, time(23, 59))
    scheduled_times = get_doctor_schedule(doctor_id, start_datetime, end_datetime)
    
    # Get patient's existing appointments for the day (with any doctor)
    patient_scheduled_times = []
    if patient_id:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            try:
                cursor.execute("""
                    SELECT appointment_date
                    FROM appointments
                    WHERE patient_id = %s 
                    AND DATE(appointment_date) = %s
                    AND status IN ('Scheduled', 'Confirmed', 'Rescheduled')
                """, (patient_id, selected_date))
                
                patient_appointments = cursor.fetchall()
                patient_scheduled_times = [appt['appointment_date'] for appt in patient_appointments]
            except Exception as e:
                print(f"Error fetching patient appointments: {e}")
            finally:
                cursor.close()
                conn.close()
    
    # Generate all possible slots
    available_slots = []
    current_time = datetime.combine(selected_date, time(start_hour, 0))
    end_time = datetime.combine(selected_date, time(end_hour, 0))
    
    while current_time < end_time:
        # Check if this slot is already booked with the doctor
        is_available = True
        for scheduled in scheduled_times:
            time_diff = abs((current_time - scheduled).total_seconds()) / 60
            if time_diff < slot_duration:  # Slot overlaps with existing appointment
                is_available = False
                break
        
        # Check if patient already has an appointment at this time with any doctor
        if is_available and patient_id:
            for patient_scheduled in patient_scheduled_times:
                time_diff = abs((current_time - patient_scheduled).total_seconds()) / 60
                if time_diff < slot_duration:  # Slot overlaps with patient's existing appointment
                    is_available = False
                    break
        
        if is_available:
            available_slots.append(current_time)
        
        # Move to next slot
        current_time += timedelta(minutes=slot_duration)
    
    return available_slots

def schedule_appointment(patient_id):
    """Handle appointment scheduling for patients with improved UI"""
    st.header("📆 Schedule an Appointment")
    
    # Get patient info
    conn = get_db_connection()
    if not conn:
        st.error("Database connection failed")
        return
    
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT full_name FROM patients WHERE patient_id = %s", (patient_id,))
    patient = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not patient:
        st.error("Patient information not found")
        return
    
    # Display appointment scheduling instructions
    st.markdown("""
    <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
        <h3 style="color: #1e3a8a;">Schedule Your Next Appointment</h3>
        <p>Please provide the following information to book your appointment:</p>
        <ul>
            <li>Select your preferred doctor</li>
            <li>Choose a convenient date</li>
            <li>Describe the reason for your visit</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Get available doctors
    doctors = get_available_doctors()
    if not doctors:
        st.warning("No doctors are currently available in the system. Please try again later.")
        return
    
    # Form for scheduling
    with st.form("appointment_form"):
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Select doctor with dropdown interface for single selection
            st.subheader("Select Doctor")
            doctor_options = {f"{doc['full_name']} ({doc['specialization']})": doc['doctor_id'] for doc in doctors}
            
            # Add a prompt option that can't be selected
            doctor_options = {"Select a doctor": ""} | doctor_options
            
            # Use selectbox for doctor selection
            selected_doctor_display = st.selectbox(
                "Choose your preferred doctor",
                options=list(doctor_options.keys()),
                index=0,
                key="doctor_select"
            )
            
            selected_doctor_id = doctor_options[selected_doctor_display]
            
            # Show warning if no doctor is selected
            if not selected_doctor_id:
                st.warning("Please select a doctor to continue")
        
        with col2:
            # Select date
            st.subheader("Select Date")
            min_date = datetime.now().date() + timedelta(days=1)
            max_date = min_date + timedelta(days=30)  # Allow booking up to 30 days in advance
            selected_date = st.date_input("Appointment Date", 
                                         min_value=min_date,
                                         max_value=max_date,
                                         value=min_date)
            
            # Display day of week for the selected date
            st.markdown(f"**Day:** {selected_date.strftime('%A')}")
        
        # Enter reason for visit
        st.subheader("Reason for Visit")
        visit_reason = st.text_area("Please describe your symptoms or reason for the appointment", 
                                   placeholder="Example: Annual check-up, Memory concerns, Follow-up appointment")
        
        # Additional notes or questions
        additional_notes = st.text_area("Additional Notes or Questions (Optional)", 
                                       placeholder="Any specific questions or concerns you'd like to discuss")
        
        # Submit button
        submit = st.form_submit_button("Check Available Times")
        
        if submit:
            if not selected_doctor_id:
                st.error("Please select a doctor")
                return
                
            if not visit_reason:
                st.error("Please provide a reason for your visit")
                return
            
            # Generate available time slots
            available_slots = generate_available_slots(selected_doctor_id, selected_date, patient_id)
            
            if not available_slots:
                st.warning("No available appointment slots for the selected date. Please try another date.")
                return
            
            # Get the doctor name without the specialty part
            doctor_name = selected_doctor_display.split(" (")[0] if "(" in selected_doctor_display else selected_doctor_display
            
            # Store form data in session state for the next step
            st.session_state.appointment_form_data = {
                'patient_id': patient_id,
                'doctor_id': selected_doctor_id,
                'doctor_name': doctor_name,
                'selected_date': selected_date,
                'visit_reason': visit_reason,
                'additional_notes': additional_notes,
                'available_slots': available_slots
            }
            st.rerun()  # Rerun to show time selection
    
    # If form has been submitted and time slots are available
    if 'appointment_form_data' in st.session_state:
        data = st.session_state.appointment_form_data
        
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
            <h3 style="color: #1e3a8a;">Select an Available Time</h3>
            <p><strong>Doctor:</strong> {0}</p>
            <p><strong>Date:</strong> {1}</p>
        </div>
        """.format(
            data['doctor_name'],
            data['selected_date'].strftime('%A, %B %d, %Y')
        ), unsafe_allow_html=True)
        
        # Format times for selection in a more visual way
        time_options = [slot.strftime("%I:%M %p") for slot in data['available_slots']]
        
        # Display time slots in a grid
        st.subheader("Available Time Slots")
        
        # Create columns for time slots (3 per row)
        time_cols = st.columns(3)
        selected_time_index = 0
        
        for i, time_str in enumerate(time_options):
            col_idx = i % 3
            with time_cols[col_idx]:
                if st.button(f"⏰ {time_str}", key=f"time_slot_{i}", use_container_width=True):
                    selected_time_index = i
                    st.session_state.selected_time_index = i
        
        # If a time is selected (tracked in session state)
        if 'selected_time_index' in st.session_state:
            selected_time_index = st.session_state.selected_time_index
            selected_time_str = time_options[selected_time_index]
            selected_datetime = data['available_slots'][selected_time_index]
            
            st.markdown(f"""
            <div style="background-color: #f0f9ff; padding: 15px; border-radius: 10px; border-left: 5px solid #3b82f6; margin: 20px 0;">
                <h4>Appointment Details</h4>
                <p><strong>Doctor:</strong> {data['doctor_name']}</p>
                <p><strong>Date:</strong> {data['selected_date'].strftime('%A, %B %d, %Y')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("✅ Confirm Appointment", type="primary", key="confirm_appointment_button"):
                # Save the appointment to database
                conn = get_db_connection()
                if not conn:
                    st.error("Database connection failed")
                    return
                
                cursor = conn.cursor(dictionary=True)
                try:
                    # First check again if the patient already has an appointment at the same time
                    # with any doctor to prevent race conditions
                    slot_start = selected_datetime
                    slot_end = slot_start + timedelta(minutes=30)
                    
                    cursor.execute("""
                        SELECT a.appointment_id, a.appointment_date, d.full_name as doctor_name
                        FROM appointments a
                        JOIN doctors d ON a.doctor_id = d.doctor_id
                        WHERE a.patient_id = %s 
                        AND a.status IN ('Scheduled', 'Confirmed', 'Rescheduled')
                        AND a.appointment_date BETWEEN %s AND %s
                    """, (data['patient_id'], slot_start - timedelta(minutes=29), slot_end))
                    
                    existing_appointments = cursor.fetchall()
                    
                    if existing_appointments:
                        # Patient already has an appointment at this time with another doctor
                        existing_appt = existing_appointments[0]
                        existing_time = existing_appt['appointment_date'].strftime('%I:%M %p')
                        existing_doctor = existing_appt['doctor_name']
                        
                        st.error(f"""
                        You already have an appointment with Dr. {existing_doctor} at {existing_time}.
                        Please select a different time or cancel your existing appointment first.
                        """)
                        return
                    
                    # Combine reason and notes
                    notes = data['visit_reason']
                    if data['additional_notes']:
                        notes += f"\n\nAdditional notes: {data['additional_notes']}"
                    
                    cursor.execute("""
                        INSERT INTO appointments
                        (patient_id, doctor_id, appointment_date, reason, status, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        data['patient_id'],
                        data['doctor_id'],
                        selected_datetime,
                        notes,
                        "Scheduled",
                        datetime.now()
                    ))
                    
                    conn.commit()
                    
                    # Clear the form data
                    del st.session_state.appointment_form_data
                    if 'selected_time_index' in st.session_state:
                        del st.session_state.selected_time_index
                    
                    # Show confirmation message with confetti effect
                    st.balloons()
                    st.success("✅ Your appointment has been successfully scheduled!")
                    
                    # Show appointment confirmation
                    st.markdown(f"""
                    <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-top: 20px;">
                        <h3 style="color: #1e3a8a;">Appointment Confirmation</h3>
                        <p><strong>Doctor:</strong> {data['doctor_name']}</p>
                        <p><strong>Date:</strong> {data['selected_date'].strftime('%A, %B %d, %Y')}</p>
                        <p><strong>Time:</strong> {selected_time_str}</p>
                        <p><strong>Reason:</strong> {data['visit_reason']}</p>
                        <hr style="margin: 15px 0; border: none; border-top: 1px solid #e5e7eb;">
                        <p>Please arrive 15 minutes before your scheduled appointment time.</p>
                        <p>If you need to cancel or reschedule, please do so at least 24 hours in advance.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error scheduling appointment: {e}")
                finally:
                    cursor.close()
                    conn.close()
            
            if st.button("❌ Cancel", key="cancel_time_selection_button"):
                # Clear the selected time
                if 'selected_time_index' in st.session_state:
                    del st.session_state.selected_time_index
                st.rerun()
        
        # Option to go back to date/doctor selection
        st.markdown("---")
        if st.button("⬅️ Back to Doctor & Date Selection", key="back_to_doctor_date_button"):
            # Clear the form data to start over
            del st.session_state.appointment_form_data
            if 'selected_time_index' in st.session_state:
                del st.session_state.selected_time_index
            st.rerun()

def view_appointments(patient_id):
    """View and manage existing appointments with improved UI"""
    st.header("📅 My Appointments")
    
    try:
        # Check if in reschedule mode
        if 'reschedule_mode' in st.session_state and st.session_state.reschedule_mode:
            reschedule_appointment_ui(patient_id, st.session_state.appointment_to_reschedule)
            return
        
        # Get appointments
        conn = get_db_connection()
        if not conn:
            st.error("Database connection failed")
            return
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT a.appointment_id, a.appointment_date,
                   a.status, a.reason,
                   d.full_name as doctor_name, d.specialization
            FROM appointments a
            JOIN doctors d ON a.doctor_id = d.doctor_id
            WHERE a.patient_id = %s
            ORDER BY a.appointment_date
        """, (patient_id,))
        
        appointments = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not appointments:
            st.info("You don't have any appointments scheduled yet.")
            
            # Add a button to schedule a new appointment
            if st.button("➕ Schedule an Appointment", key="no_appts_schedule_btn"):
                st.session_state.patient_portal_page = "schedule"
                st.rerun()
            return
        
        # Filter by upcoming/past
        today = datetime.now()
        upcoming_appointments = []
        past_appointments = []
        
        for appt in appointments:
            # Use appointment_date directly since it contains both date and time
            appt_datetime = appt['appointment_date']
            
            # Add formatted date/time for display
            appt['appt_date'] = appt_datetime.strftime('%B %d, %Y')
            appt['appt_time'] = appt_datetime.strftime('%I:%M %p')
            
            # Categorize as upcoming or past
            if appt_datetime > today and appt['status'] != 'Cancelled':
                upcoming_appointments.append(appt)
            else:
                past_appointments.append(appt)
        
        # Display upcoming appointments
        st.subheader("Upcoming Appointments")
        
        if upcoming_appointments:
            for appt in upcoming_appointments:
                # Determine status color
                if appt['status'] == 'Confirmed' or appt['status'] == 'Scheduled':
                    status_color = "#10B981"  # green
                elif appt['status'] == 'Rescheduled':
                    status_color = "#3B82F6"  # blue
                else:
                    status_color = "#6B7280"  # gray
                
                st.markdown(f"""
                <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <h3 style="margin: 0; color: #1e3a8a;">📅 {appt['appt_date']}</h3>
                            <p style="margin: 5px 0;">⏰ {appt['appt_time']}</p>
                        </div>
                        <div>
                            <span style="background-color: {status_color}; color: white; padding: 5px 10px; border-radius: 20px; font-size: 12px;">{appt['status']}</span>
                        </div>
                    </div>
                    <hr style="margin: 10px 0; border: none; border-top: 1px solid #e5e7eb;">
                    <p><strong>Doctor:</strong> {appt['doctor_name']} ({appt['specialization']})</p>
                    <p><strong>Reason:</strong> {appt['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Actions for this appointment (if scheduled)
                if appt['status'] in ['Scheduled', 'Confirmed', 'Rescheduled']:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✏️ Reschedule", key=f"reschedule_appt_{appt['appointment_id']}", help="Reschedule this appointment"):
                            # Set up reschedule mode and store appointment info
                            st.session_state.reschedule_mode = True
                            st.session_state.appointment_to_reschedule = appt
                            st.rerun()
                    with col2:
                        if st.button("❌ Cancel", key=f"cancel_appt_{appt['appointment_id']}", help="Cancel this appointment"):
                            cancel_result = cancel_appointment(appt['appointment_id'])
                            if cancel_result:
                                st.success("Appointment cancelled successfully")
                                st.rerun()
        else:
            st.info("You have no upcoming appointments.")
            
        # Display past appointments
        if past_appointments:
            st.subheader("Past Appointments")
            
            # Use a more compact view for past appointments
            for i, appt in enumerate(past_appointments):
                with st.expander(f"📅 {appt['appointment_date'].strftime('%Y-%m-%d')} - {appt['doctor_name']}"):
                    st.markdown(f"**Date & Time:** {appt['appointment_date'].strftime('%A, %B %d, %Y at %I:%M %p')}")
                    st.markdown(f"**Doctor:** {appt['doctor_name']} ({appt['specialization']})")
                    st.markdown(f"**Reason:** {appt['reason']}")
                    st.markdown(f"**Status:** {appt['status']}")
    
    except Exception as e:
        st.error(f"Error retrieving appointments: {e}")

def cancel_appointment(appointment_id):
    """Cancel an appointment"""
    conn = get_db_connection()
    if not conn:
        st.error("Database connection failed")
        return False
    
    cursor = conn.cursor()
    try:
        # Update appointment status to cancelled
        cursor.execute("""
            UPDATE appointments
            SET status = 'Cancelled', cancellation_date = %s
            WHERE appointment_id = %s
        """, (datetime.now(), appointment_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    
    except Exception as e:
        st.error(f"Error cancelling appointment: {e}")
        cursor.close()
        conn.close()
        return False

def reschedule_appointment_ui(patient_id, appointment):
    """UI for rescheduling an appointment."""
    st.header("✏️ Reschedule Appointment")
    
    # Back button
    if st.button("← Back to Appointments", key="back_to_appointments"):
        st.session_state.reschedule_mode = False
        if 'appointment_to_reschedule' in st.session_state:
            del st.session_state.appointment_to_reschedule
        st.rerun()
    
    # Display current appointment details
    st.subheader("Current Appointment Details")
    
    st.markdown(f"""
    <div style="background-color: #f3f4f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <p><strong>Doctor:</strong> {appointment['doctor_name']} ({appointment.get('specialization', 'Specialist')})</p>
        <p><strong>Date:</strong> {appointment['appt_date']}</p>
        <p><strong>Time:</strong> {appointment['appt_time']}</p>
        <p><strong>Reason:</strong> {appointment.get('reason', 'Consultation')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get doctor's ID
    conn = get_db_connection()
    if not conn:
        st.error("Database connection failed")
        return
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT doctor_id
            FROM appointments
            WHERE appointment_id = %s
        """, (appointment['appointment_id'],))
        
        result = cursor.fetchone()
        if not result:
            st.error("Could not retrieve appointment details")
            cursor.close()
            conn.close()
            return
        
        doctor_id = result['doctor_id']
        cursor.close()
    except Exception as e:
        st.error(f"Error retrieving doctor details: {e}")
        cursor.close()
        conn.close()
        return
    
    # New appointment selection
    st.subheader("New Appointment Details")
    
    # Date selection
    min_date = datetime.now().date() + timedelta(days=1)  # Start from tomorrow
    max_date = datetime.now().date() + timedelta(days=60)  # Allow booking up to 60 days ahead
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_date = st.date_input(
            "Select a new date",
            min_value=min_date,
            max_value=max_date,
            value=min_date
        )
    
    # Get available time slots for this doctor on the selected date
    available_slots = generate_available_slots(doctor_id, new_date, patient_id)
    
    # Remove slots that conflict with the doctor's existing appointments
    # except the current appointment being rescheduled
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT appointment_date
            FROM appointments
            WHERE doctor_id = %s 
            AND DATE(appointment_date) = %s
            AND status IN ('Scheduled', 'Confirmed', 'Rescheduled')
            AND appointment_id != %s
        """, (doctor_id, new_date, appointment['appointment_id']))
        
        booked_slots = cursor.fetchall()
        booked_times = [slot['appointment_date'].strftime('%H:%M') for slot in booked_slots]
        
        # Convert datetime objects to formatted strings for comparison
        formatted_available_slots = []
        datetime_slots = []
        
        for slot in available_slots:
            formatted_time = slot.strftime('%H:%M')
            if formatted_time not in booked_times:
                formatted_available_slots.append(slot.strftime('%I:%M %p'))
                datetime_slots.append(slot)
        
        # Check if the patient has other appointments on this date
        cursor.execute("""
            SELECT appointment_date
            FROM appointments
            WHERE patient_id = %s 
            AND DATE(appointment_date) = %s
            AND status IN ('Scheduled', 'Confirmed', 'Rescheduled')
            AND appointment_id != %s
        """, (patient_id, new_date, appointment['appointment_id']))
        
        patient_booked_slots = cursor.fetchall()
        patient_booked_times = [slot['appointment_date'].strftime('%H:%M') for slot in patient_booked_slots]
        
        # Filter out times where the patient already has appointments
        final_formatted_slots = []
        final_datetime_slots = []
        
        for i, formatted_time in enumerate(formatted_available_slots):
            time_24h = datetime_slots[i].strftime('%H:%M')
            if time_24h not in patient_booked_times:
                final_formatted_slots.append(formatted_time)
                final_datetime_slots.append(datetime_slots[i])
        
        # Store both the formatted strings and datetime objects
        st.session_state.available_formatted_slots = final_formatted_slots
        st.session_state.available_datetime_slots = final_datetime_slots
        
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Error checking appointment conflicts: {e}")
        cursor.close()
        conn.close()
        return
    
    with col2:
        if not hasattr(st.session_state, 'available_formatted_slots') or not st.session_state.available_formatted_slots:
            st.warning("No available time slots on this date. Please select another date.")
            new_time = None
        else:
            new_time = st.selectbox(
                "Select a new time",
                options=st.session_state.available_formatted_slots
            )
    
    # Allow adding a reason for rescheduling
    reschedule_reason = st.text_area(
        "Reason for rescheduling (optional)",
        placeholder="Please provide a reason for rescheduling your appointment..."
    )
    
    # Only show the confirm button if there are available slots
    if hasattr(st.session_state, 'available_formatted_slots') and st.session_state.available_formatted_slots:
        if st.button("Confirm Reschedule", type="primary", use_container_width=True):
            # Find the corresponding datetime object for the selected time
            selected_index = st.session_state.available_formatted_slots.index(new_time)
            combined_datetime = st.session_state.available_datetime_slots[selected_index]
            
            # Check for any conflicts with other doctors (final check before committing)
            conn = get_db_connection()
            if not conn:
                st.error("Database connection failed")
                return
            
            cursor = conn.cursor(dictionary=True)
            try:
                # Check if the patient has any other appointments at the same time with different doctors
                slot_start = combined_datetime
                slot_end = slot_start + timedelta(minutes=30)
                
                cursor.execute("""
                    SELECT a.appointment_id, a.appointment_date, d.full_name as doctor_name
                    FROM appointments a
                    JOIN doctors d ON a.doctor_id = d.doctor_id
                    WHERE a.patient_id = %s 
                    AND a.appointment_id != %s
                    AND a.status IN ('Scheduled', 'Confirmed', 'Rescheduled')
                    AND a.appointment_date BETWEEN %s AND %s
                """, (patient_id, appointment['appointment_id'], slot_start - timedelta(minutes=29), slot_end))
                
                conflicting_appointments = cursor.fetchall()
                cursor.close()
                conn.close()
                
                if conflicting_appointments:
                    # Patient already has another appointment at this time
                    conflict = conflicting_appointments[0]
                    conflict_time = conflict['appointment_date'].strftime('%I:%M %p')
                    conflict_doctor = conflict['doctor_name']
                    
                    st.error(f"""
                    This time conflicts with your existing appointment with Dr. {conflict_doctor} at {conflict_time}.
                    Please select a different time or cancel your other appointment first.
                    """)
                    return
                
                # No conflicts found, proceed with rescheduling
                result = reschedule_appointment(
                    appointment['appointment_id'], 
                    combined_datetime,
                    None,
                    reschedule_reason
                )
                
                if result:
                    st.success("Your appointment has been successfully rescheduled!")
                    
                    # Show confirmation 
                    st.markdown(f"""
                    <div style="background-color: #ecfdf5; padding: 20px; border-radius: 10px; border-left: 5px solid #10b981; margin-top: 20px;">
                        <h3 style="color: #10b981;">Appointment Rescheduled</h3>
                        <p><strong>Doctor:</strong> {appointment['doctor_name']}</p>
                        <p><strong>New Date:</strong> {new_date.strftime('%A, %B %d, %Y')}</p>
                        <p><strong>New Time:</strong> {new_time}</p>
                        <p>You will receive a confirmation email shortly.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add a return button
                    if st.button("Return to Appointments", use_container_width=True):
                        st.session_state.reschedule_mode = False
                        if 'appointment_to_reschedule' in st.session_state:
                            del st.session_state.appointment_to_reschedule
                        st.rerun()
                else:
                    st.error("Failed to reschedule appointment. Please try again later.")
            except Exception as e:
                st.error(f"Error rescheduling appointment: {e}")
    else:
        st.error("Please select a date with available time slots to continue.")

def reschedule_appointment(appointment_id, new_datetime, new_time=None, reason=None):
    """Reschedule an existing appointment"""
    conn = get_db_connection()
    if not conn:
        st.error("Database connection failed")
        return False
    
    cursor = conn.cursor()
    try:
        # Update the appointment with new date and time
        if reason:
            # Append the reschedule reason to the existing reason field instead of trying to use notes
            cursor.execute("""
                UPDATE appointments
                SET appointment_date = %s, 
                    status = 'Rescheduled',
                    reason = CONCAT(IFNULL(reason, ''), '\nRescheduled on ', %s, ': ', %s)
                WHERE appointment_id = %s
            """, (new_datetime, datetime.now().strftime('%Y-%m-%d'), reason, appointment_id))
        else:
            cursor.execute("""
                UPDATE appointments
                SET appointment_date = %s, 
                    status = 'Rescheduled'
                WHERE appointment_id = %s
            """, (new_datetime, appointment_id))
        
        conn.commit()
        success = cursor.rowcount > 0
        cursor.close()
        conn.close()
        return success
    except Exception as e:
        st.error(f"Error rescheduling appointment: {e}")
        cursor.close()
        conn.close()
        return False

def view_medical_history(patient_id):
    """View patient's medical records and test results with improved visualization"""
    st.header("📋 My Medical Records")
    
    conn = get_db_connection()
    if not conn:
        st.error("Database connection failed")
        return
    
    cursor = conn.cursor(dictionary=True)
    try:
        # Get medical records - remove doctor_id from the query as it doesn't exist
        cursor.execute("""
            SELECT record_id, diagnosis, visit_date, notes
            FROM medical_records
            WHERE patient_id = %s
            ORDER BY visit_date DESC
        """, (patient_id,))
        
        records = cursor.fetchall()
        
        # Get Alzheimer's analyses
        cursor.execute("""
            SELECT analysis_id, prediction, confidence_score, analyzed_at
            FROM alzheimers_analysis
            WHERE patient_id = %s
            ORDER BY analyzed_at DESC
        """, (patient_id,))
        
        analyses = cursor.fetchall()
        
        # Get MRI scans
        cursor.execute("""
            SELECT scan_id, scan_date, scan_type, is_processed
            FROM mri_scans
            WHERE patient_id = %s
            ORDER BY scan_date DESC
        """, (patient_id,))
        
        mri_scans = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Create tabs for different types of medical data
        tab1, tab2, tab3 = st.tabs(["👨‍⚕️ Doctor's Medical Records", "🤖 AI Assessments", "🔍 MRI Scans"])
        
        with tab1:
            if records:
                st.markdown("### Records from your healthcare provider")
                st.info("These are medical records entered by doctors during your clinic visits.")
                for i, record in enumerate(records):
                    # Create a card-like display for each record - remove doctor name display
                    with st.expander(f"🩺 {record['visit_date'].strftime('%Y-%m-%d')} - {record['diagnosis']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Diagnosis:** {record['diagnosis']}")
                            st.markdown(f"**Date:** {record['visit_date'].strftime('%Y-%m-%d')}")
                        
                        with col2:
                            st.markdown("**Doctor's Notes:**")
                            st.info(record['notes'])
            else:
                st.info("📋 No medical records available yet.")
                st.markdown("""
                <div style="background-color: #f0f9ff; padding: 15px; border-radius: 10px; border-left: 5px solid #3b82f6;">
                    <h4>Why are medical records important?</h4>
                    <p>Medical records help your healthcare providers track your health over time and provide the best care possible.</p>
                    <p>Records will appear here after your visits with doctors at our facility.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            if analyses:
                st.markdown("### AI-Generated Alzheimer's Assessments")
                st.info("These assessments are generated by our AI system based on your data and scans. They are not direct medical diagnoses from your doctor.")
                
                # Create a visual indicator of the latest analysis
                latest = analyses[0]
                pred = latest['prediction']
                conf = float(latest['confidence_score'])
                
                # Display prediction with appropriate colors and emoji
                if pred == "Demented" or "Alzheimer" in pred:
                    status_color = "#EF4444"  # red
                    emoji = "🔴"
                elif pred == "Converted" or "Mild" in pred:
                    status_color = "#F59E0B"  # orange
                    emoji = "🟠" 
                else:
                    status_color = "#10B981"  # green
                    emoji = "🟢"
                
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
                    <h3>Latest Assessment {emoji}</h3>
                    <p style="font-size: 24px; font-weight: bold; color: {status_color};">{pred}</p>
                    <p><strong>Confidence:</strong> {conf:.1%}</p>
                    <p><strong>Date:</strong> {latest['analyzed_at'].strftime('%Y-%m-%d %H:%M')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a visual timeline of assessments
                if len(analyses) > 1:
                    st.markdown("### Assessment History")
                    
                    # Prepare data for timeline visualization
                    history_data = []
                    for a in analyses:
                        value = 0
                        if a['prediction'] == "Demented" or "Alzheimer" in a['prediction']:
                            value = 3
                        elif a['prediction'] == "Converted" or "Mild" in a['prediction']:
                            value = 2
                        else:
                            value = 1
                            
                        history_data.append({
                            'Date': a['analyzed_at'],
                            'Value': value,
                            'Prediction': a['prediction']
                        })
                    
                    df = pd.DataFrame(history_data)
                    
                    # Create a line chart
                    fig = px.line(df, x='Date', y='Value', markers=True,
                                 labels={'Value': 'Status', 'Date': 'Assessment Date'},
                                 title='Alzheimer\'s Assessment History')
                    
                    fig.update_layout(
                        yaxis = dict(
                            tickmode = 'array',
                            tickvals = [1, 2, 3],
                            ticktext = ['Normal', 'Mild/Converted', 'Alzheimer\'s']
                        ),
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # List all assessments
                for analysis in analyses:
                    pred = analysis['prediction']
                    if pred == "Demented" or "Alzheimer" in pred:
                        icon = "🔴"
                    elif pred == "Converted" or "Mild" in pred:
                        icon = "🟠"
                    else:
                        icon = "🟢"
                    
                    with st.expander(f"{icon} {analysis['analyzed_at'].strftime('%Y-%m-%d')} - {pred}"):
                        st.markdown(f"**Prediction:** {pred}")
                        st.markdown(f"**Confidence:** {float(analysis['confidence_score']):.1%}")
                        st.markdown(f"**Date:** {analysis['analyzed_at'].strftime('%Y-%m-%d %H:%M')}")
                        
                        # Add recommendation based on prediction
                        if pred == "Demented" or "Alzheimer" in pred:
                            st.warning("Recommendation: Please consult with your neurologist for a comprehensive evaluation and discussion of treatment options.")
                        elif pred == "Converted" or "Mild" in pred:
                            st.info("Recommendation: Regular follow-up appointments are advised to monitor cognitive function.")
                        else:
                            st.success("Recommendation: Continue with regular check-ups as scheduled.")
            else:
                st.info("No Alzheimer's assessments available yet.")
                st.markdown("""
                <div style="background-color: #f0f9ff; padding: 15px; border-radius: 10px; border-left: 5px solid #3b82f6;">
                    <h4>About Alzheimer's Assessments</h4>
                    <p>Our advanced AI system analyzes your MRI scans to detect early signs of Alzheimer's disease.</p>
                    <p>Regular assessments can help detect changes in brain health early, when intervention is most effective.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            if mri_scans:
                # Group scans by year
                scans_by_year = {}
                for scan in mri_scans:
                    year = scan['scan_date'].year
                    if year not in scans_by_year:
                        scans_by_year[year] = []
                    scans_by_year[year].append(scan)
                
                # Display scans organized by year
                for year in sorted(scans_by_year.keys(), reverse=True):
                    st.subheader(f"{year}")
                    
                    scans_this_year = scans_by_year[year]
                    # Create columns to display multiple scans in a row
                    cols = st.columns(min(3, len(scans_this_year)))
                    
                    for i, scan in enumerate(scans_this_year):
                        col_idx = i % 3
                        with cols[col_idx]:
                            status = "✅ Processed" if scan['is_processed'] else "⏳ Pending"
                            st.markdown(f"""
                            <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 15px; height: 100%;">
                                <h4>🧠 {scan['scan_date'].strftime('%b %d')}</h4>
                                <p><strong>Type:</strong> {scan['scan_type']}</p>
                                <p><strong>Status:</strong> {status}</p>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("No MRI scans available yet.")
                st.markdown("""
                <div style="background-color: #f0f9ff; padding: 15px; border-radius: 10px; border-left: 5px solid #3b82f6;">
                    <h4>About MRI Scans</h4>
                    <p>Magnetic Resonance Imaging (MRI) creates detailed images of your brain. These scans are used for Alzheimer's diagnosis.</p>
                    <p>Your doctor may order an MRI scan during your evaluation.</p>
                </div>
                """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error retrieving medical history: {e}")

def update_profile(patient_id):
    """Update patient profile information"""
    st.write("## Update My Profile")
    
    conn = get_db_connection()
    if not conn:
        st.error("Database connection failed")
        return
    
    cursor = conn.cursor(dictionary=True)
    try:
        # Get current patient information
        cursor.execute("""
            SELECT full_name, birth_date, gender, contact_info, email, address,
                   emergency_contact, emergency_phone, allergies, medical_conditions
            FROM patients
            WHERE patient_id = %s
        """, (patient_id,))
        
        patient = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not patient:
            st.error("Patient information not found")
            return
        
        # Split full name into first and last name (assuming format is "First Last")
        name_parts = patient['full_name'].split(' ', 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""
        
        with st.form("update_profile_form"):
            st.write("### Personal Information")
            col1, col2 = st.columns(2)
            with col1:
                new_first_name = st.text_input("First Name", value=first_name)
                new_phone = st.text_input("Phone Number", value=patient['contact_info'])
                new_emergency_contact = st.text_input("Emergency Contact Name", 
                                                    value=patient['emergency_contact'] or "")
            
            with col2:
                new_last_name = st.text_input("Last Name", value=last_name)
                new_email = st.text_input("Email", value=patient['email'], disabled=True)
                new_emergency_phone = st.text_input("Emergency Contact Phone", 
                                                  value=patient['emergency_phone'] or "")
            
            # Address
            st.write("### Address")
            new_address = st.text_area("Home Address", value=patient['address'])
            
            # Medical Information
            st.write("### Medical Information")
            new_allergies = st.text_area("Allergies", value=patient['allergies'] or "")
            new_conditions = st.text_area("Pre-existing Medical Conditions", 
                                         value=patient['medical_conditions'] or "")
            
            # Password change option
            st.write("### Change Password (Optional)")
            col3, col4 = st.columns(2)
            with col3:
                new_password = st.text_input("New Password", type="password")
            with col4:
                confirm_password = st.text_input("Confirm New Password", type="password")
            
            submit = st.form_submit_button("Update Profile")
            
            if submit:
                # Validate phone format if changed
                if new_phone != patient['contact_info'] and not validate_phone(new_phone):
                    st.error("Please enter a valid phone number")
                    return
                
                # Validate emergency phone if provided
                if new_emergency_phone and not validate_phone(new_emergency_phone):
                    st.error("Please enter a valid emergency contact phone number")
                    return
                
                # Validate password if changing
                if new_password:
                    if new_password != confirm_password:
                        st.error("New passwords do not match")
                        return
                
                # Connect to database
                conn = get_db_connection()
                if not conn:
                    st.error("Database connection failed")
                    return
                
                cursor = conn.cursor()
                
                try:
                    # Update patient information
                    new_full_name = f"{new_first_name} {new_last_name}".strip()
                    
                    cursor.execute("""
                        UPDATE patients
                        SET full_name = %s, contact_info = %s, address = %s,
                            emergency_contact = %s, emergency_phone = %s,
                            allergies = %s, medical_conditions = %s
                        WHERE patient_id = %s
                    """, (
                        new_full_name, new_phone, new_address,
                        new_emergency_contact, new_emergency_phone,
                        new_allergies, new_conditions,
                        patient_id
                    ))
                    
                    # Update password if changed
                    if new_password:
                        cursor.execute("""
                            UPDATE users
                            SET password = %s
                            WHERE patient_id = %s
                        """, (hash_password(new_password), patient_id))
                    
                    conn.commit()
                    st.success("✅ Profile updated successfully!")
                    
                except Exception as e:
                    st.error(f"Error updating profile: {e}")
                finally:
                    cursor.close()
                    conn.close()
    
    except Exception as e:
        st.error(f"Error retrieving patient profile: {e}")

def patient_portal():
    """Main function for the patient portal"""
    st.title("🏥 Smart Clinic Patient Portal")
    
    # Check if user is logged in
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        # Option to register or login
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown("""
            <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h2 style="color: #1e3a8a; text-align: center;">Welcome Back! 👋</h2>
                <p style="text-align: center;">Please sign in to access your health records and services</p>
            </div>
            """, unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["🔑 Login", "✏️ Register"])
            
            with tab1:
                with st.form("login_form"):
                    email = st.text_input("📧 Email").strip()
                    password = st.text_input("🔒 Password", type="password").strip()
                    
                    try:
                        submit = st.form_submit_button("Sign In")
                        
                        if submit:
                            if email and password:
                                # Connect to database
                                conn = get_db_connection()
                                if not conn:
                                    st.error("Database connection failed")
                                    return
                                
                                cursor = conn.cursor(dictionary=True)
                                try:
                                    # First get user account
                                    cursor.execute("""
                                        SELECT id, role, patient_id
                                        FROM users
                                        WHERE username = %s AND password = %s AND role = 'patient'
                                    """, (email, hash_password(password)))
                                    
                                    user = cursor.fetchone()
                                    
                                    if user:
                                        st.session_state.logged_in = True
                                        st.session_state.user_id = user['id']
                                        st.session_state.role = user['role']
                                        st.session_state.patient_id = user['patient_id']
                                        
                                        # Get patient name
                                        cursor.execute("SELECT full_name FROM patients WHERE patient_id = %s", 
                                                    (user['patient_id'],))
                                        patient = cursor.fetchone()
                                        if patient:
                                            st.session_state.patient_name = patient['full_name']
                                        
                                        st.success("Login successful!")
                                        st.rerun()
                                    else:
                                        st.error("Invalid email or password")
                                
                                except Exception as e:
                                    st.error(f"Login error: {e}")
                                finally:
                                    cursor.close()
                                    conn.close()
                            else:
                                st.warning("Please enter both email and password")
                    except Exception as e:
                        st.error(f"Form error: {e}")
                
                # Check if user just registered
                if "temp_registered" in st.session_state and st.session_state.temp_registered:
                    st.info(f"Your account has been created with email: {st.session_state.temp_email}. Please log in now.")
                    del st.session_state.temp_registered
                    del st.session_state.temp_email
            
            with tab2:
                try:
                    register_patient()
                except Exception as e:
                    st.error(f"Registration error: {e}")
        
        with col2:
            st.markdown("""
            <div style="background-color: #1e3a8a; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); height: 100%; color: white;">
                <h2 style="color: white; text-align: center;">Smart Clinic Patient Portal</h2>
                <p style="text-align: center; margin-bottom: 30px;">Your health information at your fingertips</p>
                <div style="text-align: center;">
                    <div style="font-size: 80px; margin-bottom: 20px;">🧠</div>
                    <h3 style="color: white;">Features</h3>
                    <p>✅ Access your medical records</p>
                    <p>✅ Schedule appointments online</p>
                    <p>✅ View Alzheimer's screening results</p>
                    <p>✅ Manage your health information</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Patient is logged in - show patient dashboard
        patient_name = st.session_state.get('patient_name', 'Patient')
        
        # Sidebar for navigation - check if we're running standalone or embedded
        is_embedded = st.session_state.get("show_patient_portal", False)
        
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/brain.png", width=80)
            st.subheader(f"Welcome, {patient_name}! 👋")
            st.markdown("---")
            
            # Navigation options
            page = st.radio("", [
                "📊 Dashboard",
                "📆 Schedule Appointment",
                "🗓️ My Appointments",
                "📋 Medical Records",
                "👤 My Profile",
            ])
            
            # Only add the sign out button if we're running standalone (not embedded in app.py)
            if not is_embedded:
                st.markdown("---")
                if st.button("🚪 Sign Out", key="patient_standalone_logout"):
                    st.session_state.clear()
                    st.rerun()
        
        # Display page based on selection
        if page == "📊 Dashboard":
            display_dashboard(st.session_state.patient_id)
        elif page == "📆 Schedule Appointment":
            schedule_appointment(st.session_state.patient_id)
        elif page == "🗓️ My Appointments":
            view_appointments(st.session_state.patient_id)
        elif page == "📋 Medical Records":
            view_medical_history(st.session_state.patient_id)
        elif page == "👤 My Profile":
            update_profile(st.session_state.patient_id)

def display_dashboard(patient_id):
    """Display the patient dashboard with key metrics and information"""
    st.header("📊 Your Health Dashboard")
    
    # Get patient data from database
    conn = get_db_connection()
    if not conn:
        st.error("Database connection failed")
        return
    
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Get patient basic info
        cursor.execute("""
            SELECT full_name, birth_date, gender
            FROM patients 
            WHERE patient_id = %s
        """, (patient_id,))
        patient = cursor.fetchone()
        
        # Get upcoming appointments
        cursor.execute("""
            SELECT COUNT(*) as upcoming_count
            FROM appointments 
            WHERE patient_id = %s AND appointment_date > NOW() AND status = 'Scheduled'
        """, (patient_id,))
        upcoming = cursor.fetchone()
        
        # Get latest Alzheimer's assessment
        cursor.execute("""
            SELECT prediction, confidence_score, analyzed_at
            FROM alzheimers_analysis
            WHERE patient_id = %s
            ORDER BY analyzed_at DESC
            LIMIT 1
        """, (patient_id,))
        latest_analysis = cursor.fetchone()
        
        # Get total medical records
        cursor.execute("""
            SELECT COUNT(*) as records_count
            FROM medical_records
            WHERE patient_id = %s
        """, (patient_id,))
        records = cursor.fetchone()
        
        # Get total MRI scans
        cursor.execute("""
            SELECT COUNT(*) as scans_count
            FROM mri_scans
            WHERE patient_id = %s
        """, (patient_id,))
        scans = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        # Display patient info and metrics
        if patient:
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
                <h3>Hello, {patient['full_name']}!</h3>
                <p>Age: {calculate_age(patient['birth_date'])} • Gender: {patient['gender']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Upcoming Appointments</div>
                <div class="metric-value">🗓️ {}</div>
            </div>
            """.format(upcoming['upcoming_count'] if upcoming else 0), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Medical Records</div>
                <div class="metric-value">📋 {}</div>
            </div>
            """.format(records['records_count'] if records else 0), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">MRI Scans</div>
                <div class="metric-value">🧠 {}</div>
            </div>
            """.format(scans['scans_count'] if scans else 0), unsafe_allow_html=True)
        
        with col4:
            latest_status = "No assessment yet"
            status_emoji = "❓"
            
            if latest_analysis:
                pred = latest_analysis['prediction']
                if pred == "Demented" or "Alzheimer" in pred:
                    latest_status = "Requires Attention"
                    status_emoji = "🔴"
                elif pred == "Converted" or "Mild" in pred:
                    latest_status = "Monitor"
                    status_emoji = "🟠"
                else:
                    latest_status = "Normal"
                    status_emoji = "🟢"
            
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Current Status</div>
                <div class="metric-value">{} {}</div>
            </div>
            """.format(status_emoji, latest_status), unsafe_allow_html=True)
        
        # Show next appointment
        st.markdown("### Next Appointment")
        
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT a.appointment_date, a.reason, d.full_name as doctor_name, d.specialization
                FROM appointments a
                JOIN doctors d ON a.doctor_id = d.doctor_id
                WHERE a.patient_id = %s AND a.appointment_date > NOW() AND a.status = 'Scheduled'
                ORDER BY a.appointment_date ASC
                LIMIT 1
            """, (patient_id,))
            
            next_appt = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if next_appt:
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <h4>📅 {next_appt['appointment_date'].strftime('%A, %B %d, %Y')} at {next_appt['appointment_date'].strftime('%I:%M %p')}</h4>
                    <p><strong>Doctor:</strong> {next_appt['doctor_name']} ({next_appt['specialization']})</p>
                    <p><strong>Reason:</strong> {next_appt['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("You have no upcoming appointments. Would you like to schedule one?")
        
        # Show latest assessment results if available
        if latest_analysis:
            st.markdown("### Latest Alzheimer's Assessment")
            
            # Create a gauge chart for confidence score
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = float(latest_analysis['confidence_score']) * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#1e3a8a"},
                    'steps': [
                        {'range': [0, 33], 'color': "#10B981"},
                        {'range': [33, 66], 'color': "#F59E0B"},
                        {'range': [66, 100], 'color': "#EF4444"}
                    ],
                }
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); height: 100%;">
                    <h4>Assessment Result</h4>
                    <p><strong>Prediction:</strong> {latest_analysis['prediction']}</p>
                    <p><strong>Date:</strong> {latest_analysis['analyzed_at'].strftime('%Y-%m-%d')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")

def calculate_age(birth_date):
    """Calculate age from birth date"""
    today = datetime.now().date()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

# Run the patient portal when executed directly
if __name__ == "__main__":
    patient_portal() 
