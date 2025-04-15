import streamlit as st
import pandas as pd
import mysql.connector
from datetime import datetime, timedelta, time

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

def admin_panel():
    """Admin dashboard for managing users, doctors, patients, and logs."""
    st.set_page_config(page_title="Admin Panel", layout="wide")
    st.title("🔐 Admin Dashboard")

    # Logout button in sidebar
    if st.sidebar.button("🚪 Sign Out"):
        st.session_state.clear()
        st.success("You have been signed out.")
        st.rerun()
    
    # Connect to database
    conn = get_db_connection()
    if not conn:
        st.error("❌ Failed to connect to the database.")
        return
    cursor = conn.cursor()

    # Sidebar with stats
    with st.sidebar:
        st.subheader("📊 System Statistics")
        
        # Count users
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        # Count doctors
        cursor.execute("SELECT COUNT(*) FROM doctors")
        doctor_count = cursor.fetchone()[0]
        
        # Count patients
        cursor.execute("SELECT COUNT(*) FROM patients")
        patient_count = cursor.fetchone()[0]
        
        # Count analyses
        cursor.execute("SELECT COUNT(*) FROM alzheimers_analysis")
        analysis_count = cursor.fetchone()[0]
        
        # Display stats
        st.metric("Total Users", user_count)
        st.metric("Doctors", doctor_count)
        st.metric("Patients", patient_count)
        st.metric("Analyses", analysis_count)

    # Tabs for different admin functions
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👥 Users", "🧑‍⚕️ Doctors", "🧑‍🤝‍🧑 Patients", "🧠 Prediction Logs", "📅 Appointments"
    ])

    # USERS TAB
    with tab1:
        st.subheader("Manage Users")
        
        # Get all users
        cursor.execute("SELECT id, username, role FROM users")
        users = cursor.fetchall()
        df_users = pd.DataFrame(users, columns=["ID", "Username", "Role"])
        
        # Display users with row highlighting
        st.dataframe(df_users, use_container_width=True)

        # Two columns for Add User and Delete User
        col1, col2 = st.columns(2)
        
        # Add new user
        with col1:
            st.markdown("### ➕ Add New User")
            new_username = st.text_input("Username").strip()
            new_password = st.text_input("Password", type="password").strip()
            new_role = st.selectbox("Role", ["admin", "doctor"])
            if st.button("Add User"):
                if new_username and new_password:
                    try:
                        # Insert user without hashing password
                        cursor.execute(
                            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
                            (new_username, new_password, new_role)
                        )
                        conn.commit()
                        st.success("✅ User added successfully.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding user: {e}")
                else:
                    st.warning("Please fill in all fields.")
        
        # Delete user
        with col2:
            st.markdown("### 🗑️ Delete User")
            if not df_users.empty:
                user_to_delete = st.selectbox("Select user to delete", df_users["Username"])
                if st.button("Delete User", type="primary", help="This action cannot be undone"):
                    try:
                        cursor.execute("DELETE FROM users WHERE username = %s", (user_to_delete,))
                        conn.commit()
                        st.success(f"✅ User {user_to_delete} deleted successfully.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting user: {e}")
            else:
                st.info("No users to delete.")

    # DOCTORS TAB
    with tab2:
        st.subheader("Manage Doctors")
        
        # Get all doctors
        cursor.execute("SELECT doctor_id, full_name, specialization, email, phone_number FROM doctors")
        doctors = cursor.fetchall()
        df_doctors = pd.DataFrame(doctors, columns=[
            "ID", "Full Name", "Specialization", "Email", "Phone"
        ])
        
        # Display doctors
        st.dataframe(df_doctors, use_container_width=True)
        
        # Add doctor form
        st.markdown("### ➕ Add New Doctor")
        with st.form("add_doctor_form"):
            dr_name = st.text_input("Full Name").strip()
            dr_spec = st.text_input("Specialization").strip()
            dr_email = st.text_input("Email").strip()
            dr_phone = st.text_input("Phone").strip()
            
            submit_button = st.form_submit_button("Add Doctor")
            if submit_button:
                if dr_name and dr_spec and dr_email:
                    try:
                        cursor.execute("""
                            INSERT INTO doctors (full_name, specialization, email, phone_number)
                            VALUES (%s, %s, %s, %s)
                        """, (dr_name, dr_spec, dr_email, dr_phone))
                        conn.commit()
                        st.success("✅ Doctor added successfully.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding doctor: {e}")
                else:
                    st.warning("Please fill in all required fields.")
        
        # Delete doctor
        if not df_doctors.empty:
            st.markdown("### 🗑️ Delete Doctor")
            doctor_to_delete = st.selectbox("Select doctor to delete", 
                                         df_doctors["Full Name"])
            if st.button("Delete Doctor", help="This action cannot be undone"):
                try:
                    # Get doctor ID first
                    cursor.execute("SELECT doctor_id FROM doctors WHERE full_name = %s", 
                                (doctor_to_delete,))
                    doctor_id = cursor.fetchone()[0]
                    
                    # Delete the doctor
                    cursor.execute("DELETE FROM doctors WHERE doctor_id = %s", (doctor_id,))
                    conn.commit()
                    st.success(f"✅ Doctor {doctor_to_delete} deleted successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting doctor: {e}")

    # PATIENTS TAB
    with tab3:
        st.subheader("Patient Management")
        
        # Get all patients
        cursor.execute("""
            SELECT patient_id, full_name, gender, birth_date, contact_info, address, created_at
            FROM patients
        """)
        patients = cursor.fetchall()
        df_patients = pd.DataFrame(patients, columns=[
            "ID", "Name", "Gender", "Birthdate", "Contact", "Address", "Registered"
        ])
        
        # Display patients
        st.dataframe(df_patients, use_container_width=True)

        # Two columns for Add Patient and Patient Details
        col1, col2 = st.columns(2)
        
        # Add patient form
        with col1:
            st.markdown("### ➕ Add New Patient")
            with st.form("add_patient_form"):
                name = st.text_input("Full Name").strip()
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                birthdate = st.date_input("Date of Birth")
                contact = st.text_input("Contact Information").strip()
                address = st.text_area("Address").strip()
                
                submit_button = st.form_submit_button("Add Patient")
                if submit_button:
                    if name and contact:
                        try:
                            cursor.execute("""
                                INSERT INTO patients 
                                (full_name, gender, birth_date, contact_info, address, created_at)
                                VALUES (%s, %s, %s, %s, %s, %s)
                            """, (name, gender, birthdate, contact, address, datetime.now()))
                            conn.commit()
                            st.success("✅ Patient added successfully.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error adding patient: {e}")
                    else:
                        st.warning("Please fill in at least name and contact information.")
        
        # View patient details
        with col2:
            if not df_patients.empty:
                st.markdown("### 📋 Patient Details")
                patient_to_view = st.selectbox("Select patient to view", 
                                            df_patients["Name"].tolist())
                
                # Get selected patient ID
                patient_id = int(df_patients.loc[df_patients["Name"] == patient_to_view, "ID"].values[0])
                
                # Get medical records
                cursor.execute("""
                    SELECT diagnosis, visit_date, notes 
                    FROM medical_records 
                    WHERE patient_id = %s
                    ORDER BY visit_date DESC
                """, (patient_id,))
                records = cursor.fetchall()
                
                # Get analysis results
                cursor.execute("""
                    SELECT prediction, confidence_score, analyzed_at 
                    FROM alzheimers_analysis 
                    WHERE patient_id = %s
                    ORDER BY analyzed_at DESC
                """, (patient_id,))
                analyses = cursor.fetchall()
                
                # Display medical records
                st.markdown("#### Medical Records")
                if records:
                    df_records = pd.DataFrame(records, columns=["Diagnosis", "Visit Date", "Notes"])
                    st.dataframe(df_records, use_container_width=True)
                else:
                    st.info("No medical records found.")
                
                # Display analysis results
                st.markdown("#### Alzheimer Analysis Results")
                if analyses:
                    df_analyses = pd.DataFrame(analyses, columns=["Prediction", "Confidence", "Date"])
                    st.dataframe(df_analyses, use_container_width=True)
                else:
                    st.info("No analysis results found.")
                
                # Delete patient button
                if st.button("🗑️ Delete Patient", type="primary", help="This will delete all patient data"):
                    try:
                        # Delete patient and all related records
                        cursor.execute("DELETE FROM patients WHERE patient_id = %s", (patient_id,))
                        conn.commit()
                        st.success(f"✅ Patient {patient_to_view} deleted successfully.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting patient: {e}")

    # PREDICTION LOGS TAB
    with tab4:
        st.subheader("Alzheimer's Prediction History")
        
        # Get all analyses
        cursor.execute("""
            SELECT a.analysis_id, p.full_name, a.prediction, a.confidence_score, a.analyzed_at
            FROM alzheimers_analysis a
            JOIN patients p ON a.patient_id = p.patient_id
            ORDER BY a.analyzed_at DESC
        """)
        logs = cursor.fetchall()
        
        if logs:
            df_logs = pd.DataFrame(logs, columns=[
                "Analysis ID", "Patient Name", "Prediction", "Confidence", "Timestamp"
            ])
            
            # Add filter for predictions
            prediction_filter = st.multiselect(
                "Filter by prediction", 
                options=sorted(df_logs["Prediction"].unique())
            )
            
            if prediction_filter:
                filtered_df = df_logs[df_logs["Prediction"].isin(prediction_filter)]
            else:
                filtered_df = df_logs
            
            # Display filtered dataframe
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download CSV button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="alzheimers_predictions.csv",
                mime="text/csv",
            )
        else:
            st.info("No prediction logs found.")

    # APPOINTMENTS TAB
    with tab5:
        st.subheader("Appointment Management")
        
        # Get all appointments with patient and doctor names
        cursor.execute("""
            SELECT a.appointment_id, p.full_name, d.full_name, 
                   a.appointment_date, a.reason, a.status
            FROM appointments a
            JOIN patients p ON a.patient_id = p.patient_id
            JOIN doctors d ON a.doctor_id = d.doctor_id
            ORDER BY a.appointment_date DESC
        """)
        appointments = cursor.fetchall()
        
        if appointments:
            df_appointments = pd.DataFrame(appointments, columns=[
                "ID", "Patient", "Doctor", "Date", "Reason", "Status"
            ])
            
            # Filter by status
            status_filter = st.multiselect(
                "Filter by status",
                options=["Scheduled", "Completed", "Cancelled"],
                default=["Scheduled"]
            )
            
            if status_filter:
                filtered_appts = df_appointments[df_appointments["Status"].isin(status_filter)]
            else:
                filtered_appts = df_appointments
            
            # Display filtered appointments
            st.dataframe(filtered_appts, use_container_width=True)
            
            # Update appointment status
            st.markdown("### ✏️ Update Appointment Status")
            appt_id = st.selectbox("Select appointment", filtered_appts["ID"].tolist())
            new_status = st.selectbox("New status", ["Scheduled", "Completed", "Cancelled"])
            
            if st.button("Update Status"):
                try:
                    cursor.execute(
                        "UPDATE appointments SET status = %s WHERE appointment_id = %s",
                        (new_status, appt_id)
                    )
                    conn.commit()
                    st.success(f"✅ Appointment status updated to {new_status}.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error updating appointment: {e}")
        else:
            st.info("No appointments found.")
        
        # Add new appointment
        st.markdown("### ➕ Schedule New Appointment")
        with st.form("add_appointment_form"):
            # Get patient and doctor lists
            patient_options = {}
            doctor_options = {}
            
            try:
                cursor.execute("SELECT patient_id, full_name FROM patients")
                patients_list = cursor.fetchall()
                patient_options = {p[1]: p[0] for p in patients_list}
                
                cursor.execute("SELECT doctor_id, full_name FROM doctors")
                doctors_list = cursor.fetchall()
                doctor_options = {d[1]: d[0] for d in doctors_list}
            except Exception as e:
                st.error(f"Error fetching patients/doctors: {e}")
            
            # Form fields
            if patient_options:
                patient_name = st.selectbox("Patient", options=list(patient_options.keys()))
            else:
                st.error("No patients found. Please add patients first.")
                patient_name = None
                
            if doctor_options:
                doctor_name = st.selectbox("Doctor", options=list(doctor_options.keys()))
            else:
                st.error("No doctors found. Please add doctors first.")
                doctor_name = None
                
            appt_date = st.date_input("Appointment Date", value=datetime.now().date() + timedelta(days=1))
            appt_time = st.time_input("Appointment Time", value=time(9, 0))
            appt_datetime = datetime.combine(appt_date, appt_time)
            reason = st.text_area("Reason for Visit")
            
            # Always include the submit button, even if there are no patients/doctors
            submit_button = st.form_submit_button("Schedule Appointment")
            
            if submit_button:
                if patient_name and doctor_name and appt_datetime:
                    try:
                        patient_id = patient_options[patient_name]
                        doctor_id = doctor_options[doctor_name]
                        
                        cursor.execute("""
                            INSERT INTO appointments 
                            (patient_id, doctor_id, appointment_date, reason, status)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (patient_id, doctor_id, appt_datetime, reason, "Scheduled"))
                        conn.commit()
                        st.success("✅ Appointment scheduled successfully.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error scheduling appointment: {e}")
                else:
                    st.warning("Please complete all required fields.")

    # Close database connection
    cursor.close()
    conn.close()