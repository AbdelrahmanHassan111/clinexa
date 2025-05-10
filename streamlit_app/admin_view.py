import streamlit as st
import pandas as pd
import mysql.connector
from datetime import datetime, timedelta, time
from payment_management import manage_payments_and_invoices

# Database connection parameters
DB_CONFIG = {
   "host": st.secrets["connections.mysql"]["host"],
    "port": st.secrets[connections.mysql]["port"],
    "user": st.secrets[connections.mysql]["username"],
    "password": st.secrets[connections.mysql]["password"],
    "database": st.secrets[connections.mysql]["database"]
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
    
    # Apply custom styling
    st.markdown("""
    <style>
        /* Admin panel specific styles */
        .admin-header {
            background: linear-gradient(to right, #1e3a8a, #3b82f6);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .admin-header h1 {
            margin: 0;
            color: white;
            font-size: 2.2rem;
            border-bottom: none;
        }
        
        .metric-card {
            background-color: white;
            border-radius: 10px;
            padding: 1.2rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            border: 1px solid #e5e7eb;
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        .metric-icon {
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
            color: #3b82f6;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1e3a8a;
            margin: 0.3rem 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .form-card {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            border: 1px solid #e5e7eb;
            margin-bottom: 1.5rem;
        }
        
        .tab-content {
            padding: 1.5rem 0;
        }
        
        /* Improve table appearance */
        .dataframe {
            font-size: 0.9rem;
        }
        
        .dataframe th {
            background-color: #f1f5f9;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Custom header
    st.markdown("""
    <div class="admin-header">
        <div style="font-size: 2.5rem;">🔐</div>
        <h1>Admin Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)

    # Improved sidebar
    with st.sidebar:
        # Center logo with larger size
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image("streamlit_app/logo.png", width=120)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.title("Clinexa Admin")
        st.caption("Beyond Data. Beyond Care.")
        
        st.markdown("#### Navigation")
        st.markdown("---")
        st.markdown("### 📊 System Overview")
        
        # Connect to database
        conn = get_db_connection()
        if not conn:
            st.error("❌ Failed to connect to the database.")
            return
        cursor = conn.cursor()
        
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
        
        # Count MRI scans
        cursor.execute("SELECT COUNT(*) FROM mri_scans")
        mri_count = cursor.fetchone()[0]
        
        # Count invoices
        cursor.execute("SELECT COUNT(*) FROM invoices")
        invoice_count = cursor.fetchone()[0]
        
        # Display improved stats with icons
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">👥</div>
            <div class="metric-value">{}</div>
            <div class="metric-label">Users</div>
        </div>
        """.format(user_count), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🧑‍⚕️</div>
            <div class="metric-value">{}</div>
            <div class="metric-label">Doctors</div>
        </div>
        """.format(doctor_count), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🧑‍🤝‍🧑</div>
            <div class="metric-value">{}</div>
            <div class="metric-label">Patients</div>
        </div>
        """.format(patient_count), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🧠</div>
            <div class="metric-value">{}</div>
            <div class="metric-label">Analyses</div>
        </div>
        """.format(analysis_count), unsafe_allow_html=True)
        
        # Divider
        st.markdown("---")
        
        # Sign out button with improved styling
        if st.button("🚪 Sign Out", use_container_width=True, type="primary", key="admin_view_logout"):
            st.session_state.clear()
            st.success("You have been signed out.")
            st.rerun()

    # Tabs for different admin functions
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "👥 Users", "🧑‍⚕️ Doctors", "🧑‍🤝‍🧑 Patients", "🧠 Prediction Logs", "📅 Appointments", "💰 Payments"
    ])

    # USERS TAB
    with tab1:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <div style="font-size: 1.8rem; margin-right: 0.5rem;">👥</div>
            <h2 style="margin: 0;">User Management</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Get all users
        cursor.execute("SELECT id, username, role FROM users")
        users = cursor.fetchall()
        
        # Create a dataframe with better structure
        df_users = pd.DataFrame(users, columns=["ID", "Username", "Role"])
        
        # Apply styling to roles
        def highlight_role(role):
            if role == 'admin':
                return 'background-color: #e0f2fe; color: #0369a1; font-weight: 600; padding: 2px 8px; border-radius: 4px;'
            elif role == 'doctor':
                return 'background-color: #dcfce7; color: #166534; font-weight: 600; padding: 2px 8px; border-radius: 4px;'
            return ''
        
        # Style the dataframe
        styled_df = df_users.style.applymap(highlight_role, subset=['Role'])
        
        # Display users with enhanced styling
        st.dataframe(styled_df, use_container_width=True, height=300)
        
        # User management actions in a visually appealing card layout
        st.markdown("<div class='form-card'>", unsafe_allow_html=True)
        
        # Two columns for Add User and Delete User
        col1, col2 = st.columns(2)
        
        # Add new user with improved form
        with col1:
            st.markdown("""
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="font-size: 1.4rem; margin-right: 0.5rem; color: #3b82f6;">➕</div>
                <h3 style="margin: 0; color: #1e3a8a;">Add New User</h3>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form(key="add_user_form"):
                new_username = st.text_input("Username", key="new_username", placeholder="Enter username").strip()
                new_password = st.text_input("Password", type="password", key="new_password", placeholder="Enter secure password").strip()
                new_role = st.selectbox("Role", ["admin", "doctor"], key="new_role")
                
                # Description of roles
                st.markdown("""
                <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #6b7280;">
                    <strong>Admin:</strong> Full system access<br>
                    <strong>Doctor:</strong> Patient management and diagnosis access
                </div>
                """, unsafe_allow_html=True)
                
                add_user_submit = st.form_submit_button("Add User", type="primary", use_container_width=True)
                if add_user_submit:
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
                        st.warning("⚠️ Please fill in all fields.")
        
        # Delete user with improved UI
        with col2:
            st.markdown("""
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="font-size: 1.4rem; margin-right: 0.5rem; color: #ef4444;">🗑️</div>
                <h3 style="margin: 0; color: #1e3a8a;">Delete User</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if not df_users.empty:
                with st.form(key="delete_user_form"):
                    user_to_delete = st.selectbox("Select user to delete", df_users["Username"], key="user_to_delete")
                    
                    # Warning message
                    st.markdown("""
                    <div style="background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 0.8rem; margin: 1rem 0; border-radius: 4px;">
                        <div style="font-weight: 600; color: #b91c1c; margin-bottom: 0.3rem;">⚠️ Warning</div>
                        <div style="color: #7f1d1d; font-size: 0.9rem;">This action cannot be undone. All user data will be permanently deleted.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    delete_user_submit = st.form_submit_button("Delete User", use_container_width=True)
                    if delete_user_submit:
                        try:
                            cursor.execute("DELETE FROM users WHERE username = %s", (user_to_delete,))
                            conn.commit()
                            st.success(f"✅ User {user_to_delete} deleted successfully.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting user: {e}")
            else:
                st.info("No users to delete.")
                
        st.markdown("</div>", unsafe_allow_html=True)

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
            
            with st.form(key="delete_doctor_form"):
                doctor_to_delete = st.selectbox(
                    "Select doctor to delete", 
                    df_doctors["Full Name"],
                    key="doctor_to_delete"
                )
                
                delete_doctor_submit = st.form_submit_button("Delete Doctor", help="This action cannot be undone")
                if delete_doctor_submit:
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
            SELECT patient_id, full_name, gender, birth_date, contact_info, address
            FROM patients
        """)
        patients = cursor.fetchall()
        df_patients = pd.DataFrame(patients, columns=[
            "ID", "Name", "Gender", "Birthdate", "Contact", "Address"
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
                                (full_name, gender, birth_date, contact_info, address)
                                VALUES (%s, %s, %s, %s, %s)
                            """, (name, gender, birthdate, contact, address))
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
                
                # Get MRI scans
                cursor.execute("""
                    SELECT scan_type, scan_date, is_processed
                    FROM mri_scans
                    WHERE patient_id = %s
                    ORDER BY scan_date DESC
                """, (patient_id,))
                mri_scans = cursor.fetchall()
                
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
                
                # Display MRI scans
                st.markdown("#### MRI Scans")
                if mri_scans:
                    df_mri = pd.DataFrame(mri_scans, columns=["Type", "Date", "Processed"])
                    st.dataframe(df_mri, use_container_width=True)
                else:
                    st.info("No MRI scans found.")
                
                # Delete patient button
                with st.form(key="delete_patient_form"):
                    st.warning("This will delete the patient and all associated data!")
                    delete_patient_submit = st.form_submit_button("🗑️ Delete Patient", help="This will delete all patient data")
                    if delete_patient_submit:
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
            
            # Create a form for updating the appointment status
            with st.form(key="update_appointment_form"):
                new_status = st.selectbox(
                    "New status", 
                    ["Scheduled", "Completed", "Cancelled"],
                    key="new_status_selectbox"
                )
                
                update_status_submit = st.form_submit_button("Update Status")
                if update_status_submit:
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
    
    # PAYMENTS AND INVOICES TAB
    with tab6:
        manage_payments_and_invoices()

    # Close database connection
    cursor.close()
    conn.close()
