import streamlit as st
import pandas as pd
import numpy as np
import joblib
import mysql.connector
from datetime import datetime, timedelta, time
import time as time_module
import json
import google.generativeai as genai
from streamlit_chat import message
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import os

# Gemini API setup
API_KEY = "AIzaSyDIST7Xvjns3VFMf2jbawPSX95cIhAkFhA"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")

# Load the ML model - use relative path
model_path = "model/XGBoost_grid_optimized.joblib"
try:
    # Check if model file exists
    if os.path.exists(model_path):
        clf = joblib.load(model_path)
    else:
        st.warning(f"Model file not found at {model_path}. Please ensure the model file exists.")
        clf = None
except Exception as e:
    st.error(f"Error loading model: {e}")
    clf = None

# Database connection parameters
DB_CONFIG = {
    "host": "localhost",
    "user": "root", 
    "password": "root",
    "database": "smart_clinic"
}

# Direct DB Connection
def get_db_connection():
    """Create a direct database connection."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except mysql.connector.Error as e:
        st.error(f"Database connection error: {e}")
        return None

# Get the feature names required by the model
def get_feature_columns():
    """Return ordered list of feature columns required by the model"""
    return [
        "CDRSB", "mPACCdigit", "MMSE", "LDELTOTAL", "mPACCtrailsB", 
        "EcogSPPlan", "RAVLT_immediate", "FAQ", "EcogPtTotal", "Entorhinal", 
        "PTEDUCAT", "Ventricles", "Fusiform", "EcogSPOrgan", "APOE4", 
        "EcogPtLang", "FDG", "MidTemp", "TRABSCOR", "EcogSPDivatt", 
        "ADAS11", "EcogPtVisspat", "AGE", "ADAS13", "EcogSPMem", 
        "EcogPtOrgan", "ICV", "Hippocampus", "EcogSPVisspat", "MOCA", 
        "WholeBrain", "PTRACCAT", "RAVLT_learning", "DIGITSCOR", 
        "PTGENDER", "EcogSPTotal", "RAVLT_perc_forgetting", "ABETA", 
        "ADASQ4", "EcogSPLang", "EcogPtMem", "EcogPtDivatt", "RAVLT_forgetting"
    ]

# Get feature descriptions for tooltips
def get_feature_descriptions():
    return {
        "CDRSB": "Clinical Dementia Rating Sum of Boxes (Scale: 0-18, higher values indicate greater impairment)",
        "mPACCdigit": "Modified Preclinical Alzheimer's Cognitive Composite with Digit Symbol Substitution",
        "MMSE": "Mini-Mental State Examination (Scale: 0-30, higher is better)",
        "LDELTOTAL": "Logical Memory delayed recall total score",
        "mPACCtrailsB": "Modified Preclinical Alzheimer's Cognitive Composite with Trail Making Test Part B",
        "EcogSPPlan": "Study Partner Everyday Cognition Planning Score",
        "RAVLT_immediate": "Rey Auditory Verbal Learning Test - Immediate Recall score",
        "FAQ": "Functional Activities Questionnaire (Scale: 0-30, higher values indicate greater impairment)",
        "EcogPtTotal": "Patient Everyday Cognition Total Score",
        "Entorhinal": "Entorhinal cortex volume (mm³)",
        "PTEDUCAT": "Years of education",
        "Ventricles": "Ventricular volume (mm³)",
        "Fusiform": "Fusiform gyrus volume (mm³)",
        "EcogSPOrgan": "Study Partner Everyday Cognition Organization Score",
        "APOE4": "Number of APOE e4 alleles (0, 1, or 2)",
        "EcogPtLang": "Patient Everyday Cognition Language Score",
        "FDG": "Fluorodeoxyglucose (18F) PET measurement",
        "MidTemp": "Middle temporal gyrus volume (mm³)",
        "TRABSCOR": "Trail Making Test Part B score (seconds)",
        "EcogSPDivatt": "Study Partner Everyday Cognition Divided Attention Score",
        "ADAS11": "Alzheimer's Disease Assessment Scale-Cognitive Subscale (11-item version)",
        "EcogPtVisspat": "Patient Everyday Cognition Visuospatial Score",
        "AGE": "Age in years",
        "ADAS13": "Alzheimer's Disease Assessment Scale-Cognitive Subscale (13-item version)",
        "EcogSPMem": "Study Partner Everyday Cognition Memory Score",
        "EcogPtOrgan": "Patient Everyday Cognition Organization Score",
        "ICV": "Intracranial volume (mm³)",
        "Hippocampus": "Hippocampal volume (mm³)",
        "EcogSPVisspat": "Study Partner Everyday Cognition Visuospatial Score",
        "MOCA": "Montreal Cognitive Assessment (Scale: 0-30, higher is better)",
        "WholeBrain": "Whole brain volume (mm³)",
        "PTRACCAT": "Participant race category (1=White, 2=Black, 3=Asian, 4=More than one, 5=Other)",
        "RAVLT_learning": "Rey Auditory Verbal Learning Test - Learning score",
        "DIGITSCOR": "Digest Symbol Substitution Test score",
        "PTGENDER": "Participant gender (0=Male, 1=Female)",
        "EcogSPTotal": "Study Partner Everyday Cognition Total Score",
        "RAVLT_perc_forgetting": "Rey Auditory Verbal Learning Test - Percent Forgetting",
        "ABETA": "Amyloid-β levels (pg/mL)",
        "ADASQ4": "ADAS-Cog Word Finding Difficulty score",
        "EcogSPLang": "Study Partner Everyday Cognition Language Score",
        "EcogPtMem": "Patient Everyday Cognition Memory Score",
        "EcogPtDivatt": "Patient Everyday Cognition Divided Attention Score",
        "RAVLT_forgetting": "Rey Auditory Verbal Learning Test - Forgetting score",
        "EcogPtPlan": "Patient Everyday Cognition Planning Score",
        "PTMARRY": "Marital status (1=Married, 2=Widowed, 3=Divorced, 4=Never married, 5=Unknown)",
        "PTETHCAT": "Ethnicity category (1=Hispanic/Latino, 2=Not Hispanic/Latino, 3=Unknown)",
        "PTAU": "Phosphorylated tau (p-tau) levels (pg/mL)",
        "TAU": "Total tau protein levels (pg/mL)"
    }

# Get existing patient features or None
def get_patient_features(patient_id):
    conn = get_db_connection()
    if not conn:
        return None
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM alzheimer_features WHERE patient_id = %s", (patient_id,))
        features = cursor.fetchone()
        cursor.close()
        conn.close()
        return features
    except mysql.connector.Error as e:
        st.error(f"Error fetching patient features: {e}")
        cursor.close()
        conn.close()
        return None

# Prediction function
def predict_alzheimer(input_data):
    if clf is None:
        return "Error", 0
    
    try:
        # Get feature columns in the correct order
        feature_columns = get_feature_columns()
        
        # Create a properly ordered feature array with exactly 43 features
        feature_array = np.zeros(43)
        for i, feature in enumerate(feature_columns):
            if feature in input_data:
                feature_array[i] = input_data[feature]
        
        # Log the features being used for prediction
        st.session_state.last_feature_array = feature_array
                
        # Make prediction using the model
        prediction = clf.predict([feature_array])[0]
        probabilities = clf.predict_proba([feature_array])[0]
        confidence = max(probabilities)
        
        # Store all probabilities in session state for visualization
        st.session_state.last_probabilities = probabilities
        st.session_state.last_prediction_classes = clf.classes_
        
        return prediction, confidence
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return "Error", 0

# Store features in database
def store_features(patient_id, feature_data):
    conn = get_db_connection()
    if not conn:
        return False
    
    cursor = conn.cursor()
    try:
        # Check if patient already has features
        cursor.execute("SELECT feature_id FROM alzheimer_features WHERE patient_id = %s", (patient_id,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing features
            set_clause = ", ".join([f"{key} = %s" for key in feature_data.keys()])
            values = list(feature_data.values())
            values.append(patient_id)  # For WHERE clause
            
            cursor.execute(f"UPDATE alzheimer_features SET {set_clause} WHERE patient_id = %s", values)
            conn.commit()
            cursor.close()
            conn.close()
            return True
        else:
            # Insert new features
            columns = ", ".join(["patient_id"] + list(feature_data.keys()))
            placeholders = ", ".join(["%s"] * (len(feature_data) + 1))
            values = [patient_id] + list(feature_data.values())
            
            cursor.execute(f"INSERT INTO alzheimer_features ({columns}) VALUES ({placeholders})", values)
            conn.commit()
            cursor.close()
            conn.close()
            return True
    except mysql.connector.Error as e:
        st.error(f"Error storing features: {e}")
        cursor.close()
        conn.close()
        return False

# Store prediction in database
def store_prediction(patient_id, features, prediction, confidence):
    conn = get_db_connection()
    if not conn:
        return False
    
    cursor = conn.cursor()
    try:
        now = datetime.now()
        today = now.date()
        
        # Convert any numpy values to standard Python types
        converted_features = {}
        for key, value in features.items():
            if hasattr(value, "dtype"):  # Check if it's a numpy type
                converted_features[key] = value.item()  # Convert numpy value to Python scalar
            else:
                converted_features[key] = value
                
        features_json = json.dumps(converted_features)
        
        # Make sure prediction and confidence are also standard Python types
        prediction = str(prediction)
        confidence = float(confidence)
        
        # Check if a prediction already exists for this patient today
        cursor.execute("""
            SELECT analysis_id FROM alzheimers_analysis 
            WHERE patient_id = %s AND DATE(analyzed_at) = %s
        """, (patient_id, today))
        
        existing_analysis = cursor.fetchone()
        
        # Make sure to consume all results
        while cursor.fetchone() is not None:
            pass
        
        if existing_analysis:
            # Update existing prediction
            analysis_id = existing_analysis[0]
            cursor.execute("""
                UPDATE alzheimers_analysis 
                SET input_features = %s, prediction = %s, confidence_score = %s, analyzed_at = %s
                WHERE analysis_id = %s
            """, (features_json, prediction, confidence, now, analysis_id))
            
            conn.commit()
            st.info(f"Updated existing prediction from today (ID: {analysis_id}).")
            return analysis_id
        else:
            # Insert new prediction
            cursor.execute("""
                INSERT INTO alzheimers_analysis 
                (patient_id, input_features, prediction, confidence_score, analyzed_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (patient_id, features_json, prediction, confidence, now))
            
            conn.commit()
            analysis_id = cursor.lastrowid  # Get the ID of the newly inserted analysis
            return analysis_id
    except mysql.connector.Error as e:
        st.error(f"Error storing prediction: {e}")
        return False
    finally:
        # Always close cursor and connection in finally block
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Get patient medical records
def get_patient_records(patient_id):
    conn = get_db_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT record_id, diagnosis, visit_date, notes
            FROM medical_records
            WHERE patient_id = %s
            ORDER BY visit_date DESC
        """, (patient_id,))
        
        records = cursor.fetchall()
        cursor.close()
        conn.close()
        return records
    except mysql.connector.Error as e:
        st.error(f"Error fetching medical records: {e}")
        cursor.close()
        conn.close()
        return []

# Get patient personal information
def get_patient_info(patient_id):
    conn = get_db_connection()
    if not conn:
        return None
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT * FROM patients WHERE patient_id = %s
        """, (patient_id,))
        
        patient_info = cursor.fetchone()
        cursor.close()
        conn.close()
        return patient_info
    except mysql.connector.Error as e:
        st.error(f"Error fetching patient info: {e}")
        cursor.close()
        conn.close()
        return None

# Add medical record for patient
def add_medical_record(patient_id, diagnosis, notes):
    conn = get_db_connection()
    if not conn:
        return False
    
    cursor = conn.cursor()
    try:
        visit_date = datetime.now().date()
        
        cursor.execute("""
            INSERT INTO medical_records (patient_id, diagnosis, visit_date, notes)
            VALUES (%s, %s, %s, %s)
        """, (patient_id, diagnosis, visit_date, notes))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except mysql.connector.Error as e:
        st.error(f"Error adding medical record: {e}")
        cursor.close()
        conn.close()
        return False

# Get previous analyses for a patient
def get_patient_analyses(patient_id):
    conn = get_db_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT analysis_id, prediction, confidence_score, analyzed_at
            FROM alzheimers_analysis
            WHERE patient_id = %s
            ORDER BY analyzed_at DESC
        """, (patient_id,))
        
        analyses = cursor.fetchall()
        cursor.close()
        conn.close()
        return analyses
    except mysql.connector.Error as e:
        st.error(f"Error fetching analyses: {e}")
        cursor.close()
        conn.close()
        return []

# Save chat message
def save_chat_message(patient_id, doctor_id, message, sender):
    conn = get_db_connection()
    if not conn:
        return False
    
    cursor = conn.cursor()
    try:
        # First check if the doctor_id exists in the doctors table
        cursor.execute("SELECT COUNT(*) FROM doctors WHERE doctor_id = %s", (doctor_id,))
        doctor_exists = cursor.fetchone()[0] > 0
        
        # If doctor does not exist, we need to create a doctor entry using the user's ID
        if not doctor_exists:
            # Get user info
            cursor.execute("SELECT username FROM users WHERE id = %s", (doctor_id,))
            user_result = cursor.fetchone()
            
            if user_result:
                username = user_result[0]
                # Create a doctor entry with the same ID as the user
                try:
                    cursor.execute("""
                        INSERT INTO doctors (doctor_id, full_name, specialization, email, phone_number)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (doctor_id, username, "General Practitioner", f"{username}@clinic.com", "N/A"))
                    conn.commit()
                    st.info(f"Created doctor profile for user {username}")
                except mysql.connector.Error:
                    # If we can't create a doctor with the same ID, return error
                    st.error("Cannot save message: Doctor profile does not exist")
                    cursor.close()
                    conn.close()
                    return False
            else:
                st.error("User not found")
                cursor.close()
                conn.close()
                return False
                
        # Now we can insert the chat message
        now = datetime.now()
        
        # Convert sender from 'Doctor'/'Assistant' to 'doctor'/'model' for database
        db_sender = 'doctor' if sender == 'Doctor' else 'model'
        
        cursor.execute("""
            INSERT INTO chat_logs (patient_id, doctor_id, message, sender, timestamp)
            VALUES (%s, %s, %s, %s, %s)
        """, (patient_id, doctor_id, message, db_sender, now))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except mysql.connector.Error as e:
        st.error(f"Error saving chat message: {e}")
        cursor.close()
        conn.close()
        return False

# Get previous chat history
def get_chat_history(patient_id, doctor_id):
    conn = get_db_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT message, sender, timestamp
            FROM chat_logs
            WHERE patient_id = %s AND doctor_id = %s
            ORDER BY timestamp ASC
        """, (patient_id, doctor_id))
        
        chat_logs = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Convert to format expected by session state
        history = []
        for log in chat_logs:
            # Convert from database 'doctor'/'model' to UI 'You'/'Assistant'
            sender_name = "You" if log["sender"] == "doctor" else "Assistant"
            history.append((sender_name, log["message"]))
        
        return history
    except mysql.connector.Error as e:
        st.error(f"Error fetching chat history: {e}")
        cursor.close()
        conn.close()
        return []

# Generate plot of feature importance
def generate_feature_importance_plot():
    if not hasattr(clf, 'feature_importances_'):
        return None
    
    features = get_feature_columns()
    importances = clf.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Take top 15 most important features
    top_indices = indices[:15]
    top_features = [features[i] for i in top_indices]
    top_importances = [importances[i] for i in top_indices]
    
    # Create plot with extremely small size
    plt.figure(figsize=(3, 1.5))
    plt.title('Top 15 Features', fontsize=6)
    plt.barh(range(len(top_indices)), top_importances, align='center')
    plt.yticks(range(len(top_indices)), top_features, fontsize=4)
    plt.xlabel('Importance', fontsize=5)
    
    # Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

# Doctor panel main content
def doctor_panel():
    # Set page configuration
    st.set_page_config(page_title="Doctor Dashboard", layout="wide")
    
    # Initialize doctor ID from session state
    doctor_id = st.session_state.get("user_id", 1)
    
    # Sidebar menu
    with st.sidebar:
        st.title("🧑‍⚕️ Doctor Dashboard")
        
        # Navigation
        page = st.radio("Navigation", [
            "👨‍👩‍👧‍👦 Patient Management",
            "🧠 Alzheimer's Analysis",
            "📋 Medical Records",
            "💬 AI Assistant",
            "📊 Analytics"
        ])
        
        # Logout button
        if st.button("🚪 Sign Out"):
            st.session_state.clear()
            st.success("You have been signed out.")
            st.rerun()
    
    # Main content area
    conn = get_db_connection()
    if not conn:
        st.error("Could not connect to database")
        return
    
    cursor = conn.cursor()

    # Patient selection
    st.subheader("Patient Selection")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search Patient by Name or ID")
    
    with col2:
        add_new = st.button("➕ Add New Patient")
    
    if add_new:
        st.session_state.show_add_patient = True
    
    # Display add patient form if flag is set
    if st.session_state.get("show_add_patient", False):
        with st.expander("Add New Patient", expanded=True):
            added = add_patient()
            if added:
                st.session_state.show_add_patient = False
                st.rerun()
            
            if st.button("Cancel"):
                st.session_state.show_add_patient = False
                st.rerun()
    
    # Get patient list based on search term
    if search_term:
        try:
            cursor.execute("""
                SELECT patient_id, full_name, birth_date, gender, contact_info FROM patients
                WHERE full_name LIKE %s OR patient_id LIKE %s
            """, (f"%{search_term}%", f"%{search_term}%"))
            patients = cursor.fetchall()
        except mysql.connector.Error as e:
            st.error(f"Error searching patients: {e}")
            patients = []
    else:
        try:
            cursor.execute("SELECT patient_id, full_name, birth_date, gender, contact_info FROM patients ORDER BY full_name")
            patients = cursor.fetchall()
        except mysql.connector.Error as e:
            st.error(f"Error fetching patients: {e}")
            patients = []

    if not patients:
        st.warning("No patients found. Please add patients first.")
        cursor.close()
        conn.close()
        return
    
    # Create patient selection cards
    st.markdown("### Select a Patient to Continue")
    patient_cols = st.columns(3)
    
    patient_dict = {}
    for i, (pid, name, birthdate, gender, contact) in enumerate(patients):
        col = patient_cols[i % 3]
        with col:
            age = datetime.now().year - birthdate.year if birthdate else "N/A"
            card = f"""
            <div style="padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin: 5px 0;">
                <h4>{name}</h4>
                <p>ID: {pid} | Age: {age} | Gender: {gender}</p>
                <p>Contact: {contact or 'N/A'}</p>
            </div>
            """
            st.markdown(card, unsafe_allow_html=True)
            patient_dict[name] = pid
            if st.button(f"Select {name}", key=f"btn_{pid}"):
                st.session_state.selected_patient = pid
                st.session_state.selected_patient_name = name
                st.rerun()
    
    # If no patient is selected, stop here
    if "selected_patient" not in st.session_state:
        cursor.close()
        conn.close()
        return
    
    # Get selected patient ID and information
    patient_id = st.session_state.selected_patient
    patient_info = get_patient_info(patient_id)
    
    if not patient_info:
        st.error("Error retrieving patient information.")
        return
    
    # Display patient header
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
        <h2>{patient_info['full_name']}</h2>
        <p>Patient ID: {patient_id} | Date of Birth: {patient_info['birth_date']} | Gender: {patient_info['gender']}</p>
        <p>Contact: {patient_info['contact_info'] or 'N/A'} | Address: {patient_info['address'] or 'N/A'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show different content based on selected page
    if page == "👨‍👩‍👧‍👦 Patient Management":
        display_patient_management(patient_id, patient_info)
    elif page == "🧠 Alzheimer's Analysis":
        display_alzheimer_analysis(patient_id, patient_info)
    elif page == "📋 Medical Records":
        display_medical_records(patient_id, patient_info)
    elif page == "💬 AI Assistant":
        display_ai_assistant(patient_id, patient_info, doctor_id)
    elif page == "📊 Analytics":
        display_analytics(patient_id, patient_info)
    
    # Close database connection
    cursor.close()
    conn.close()

# Patient management functions
def add_patient():
    st.subheader("Add New Patient")
    
    # Patient form
    with st.form("patient_form"):
        full_name = st.text_input("Full Name")
        date_of_birth = st.date_input("Date of Birth")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        contact_number = st.text_input("Contact Number")
        email = st.text_input("Email Address")
        address = st.text_area("Address")
        
        # Form submission
        submit = st.form_submit_button("Register Patient")
        
        if submit:
            if full_name and date_of_birth and gender:
                # Connect to database
                conn = get_db_connection()
                if not conn:
                    st.error("Could not connect to database")
                    return
                
                cursor = conn.cursor()
                try:
                    # Insert patient data
                    cursor.execute("""
                        INSERT INTO patients 
                        (full_name, birth_date, gender, contact_info, email, address, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (full_name, date_of_birth, gender, contact_number, email, address, datetime.now()))
                    
                    conn.commit()
                    st.success(f"✅ Patient {full_name} registered successfully!")
                    
                    # Get the newly created patient ID for redirecting
                    patient_id = cursor.lastrowid
                    if patient_id:
                        st.session_state.selected_patient = patient_id
                        st.session_state.selected_patient_name = full_name
                    
                    cursor.close()
                    conn.close()
                    return True
                except mysql.connector.Error as e:
                    st.error(f"Error registering patient: {e}")
                    cursor.close()
                    conn.close()
                    return False
            else:
                st.warning("Please fill in all required fields (Name, Date of Birth, Gender)")
                return False
    
    return False

def view_patients():
    st.subheader("View All Patients")
    
    # Connect to database
    conn = get_db_connection()
    if not conn:
        st.error("Could not connect to database")
        return
    
    cursor = conn.cursor(dictionary=True)
    try:
        # Fetch all patients
        cursor.execute("""
            SELECT patient_id, full_name, birth_date, gender, contact_info, 
                   email, created_at 
            FROM patients
            ORDER BY created_at DESC
        """)
        
        patients = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not patients:
            st.info("No patients registered yet.")
            return
        
        # Convert to DataFrame for display
        df = pd.DataFrame(patients)
        
        # Calculate age
        df['age'] = df['birth_date'].apply(lambda x: datetime.now().year - x.year if x else 0)
        
        # Display DataFrame as a table
        st.dataframe(df, use_container_width=True)
        
        # Optionally, allow users to filter or search by name or other details
        search_term = st.text_input("Search Patients by Name")
        if search_term:
            df_filtered = df[df['full_name'].str.contains(search_term, case=False, na=False)]
            if df_filtered.empty:
                st.info(f"No patients found with name: {search_term}")
            else:
                st.dataframe(df_filtered, use_container_width=True)
        
        # Optionally, add a button to select a patient
        patient_id = st.number_input("Enter Patient ID to Select", min_value=1, step=1)
        if patient_id and patient_id in df['patient_id'].values:
            patient_info = df[df['patient_id'] == patient_id].iloc[0]
            if st.button(f"Select Patient: {patient_info['full_name']}"):
                st.session_state.selected_patient = patient_id
                st.session_state.selected_patient_name = patient_info['full_name']
                st.rerun()
                
    except Exception as e:
        st.error(f"An error occurred while fetching patients: {e}")
        cursor.close()
        conn.close()

# Display patient management page
def display_patient_management(patient_id, patient_info):
    st.header("Patient Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Details")
        st.markdown(f"""
        - **Full Name**: {patient_info['full_name']}
        - **Birth Date**: {patient_info['birth_date']}
        - **Age**: {datetime.now().year - patient_info['birth_date'].year}
        - **Gender**: {patient_info['gender']}
        - **Contact**: {patient_info['contact_info'] or 'Not provided'}
        - **Address**: {patient_info['address'] or 'Not provided'}
        - **Registration Date**: {patient_info['created_at']}
        """)
        
        # Edit patient information
        with st.expander("Edit Patient Information"):
            with st.form("edit_patient_form"):
                full_name = st.text_input("Full Name", value=patient_info['full_name'])
                birth_date = st.date_input("Birth Date", value=patient_info['birth_date'])
                gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(patient_info['gender']))
                contact = st.text_input("Contact Information", value=patient_info['contact_info'] or "")
                address = st.text_area("Address", value=patient_info['address'] or "")
                
                submit = st.form_submit_button("Update Patient Information")
                
                if submit:
                    conn = get_db_connection()
                    if conn:
                        cursor = conn.cursor()
                        try:
                            cursor.execute("""
                                UPDATE patients
                                SET full_name = %s, birth_date = %s, gender = %s, 
                                    contact_info = %s, address = %s
                                WHERE patient_id = %s
                            """, (full_name, birth_date, gender, contact, address, patient_id))
                            
                            conn.commit()
                            st.success("✅ Patient information updated successfully.")
                            
                            # Update session state
                            st.session_state.selected_patient_name = full_name
                            
                            # Refresh the page
                            time_module.sleep(1)
                            st.rerun()
                        except mysql.connector.Error as e:
                            st.error(f"Error updating patient information: {e}")
                        finally:
                            cursor.close()
                            conn.close()
    
    with col2:
        # Summary statistics
        st.subheader("Patient Summary")
        
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            
            # Count medical records
            cursor.execute("SELECT COUNT(*) FROM medical_records WHERE patient_id = %s", (patient_id,))
            record_count = cursor.fetchone()[0]
            
            # Count Alzheimer's analyses
            cursor.execute("SELECT COUNT(*) FROM alzheimers_analysis WHERE patient_id = %s", (patient_id,))
            analysis_count = cursor.fetchone()[0]
            
            # Get most recent analysis
            cursor.execute("""
                SELECT prediction, confidence_score, analyzed_at 
                FROM alzheimers_analysis 
                WHERE patient_id = %s 
                ORDER BY analyzed_at DESC LIMIT 1
            """, (patient_id,))
            latest_analysis = cursor.fetchone()
            
            # Count appointments
            cursor.execute("""
                SELECT COUNT(*) FROM appointments 
                WHERE patient_id = %s
            """, (patient_id,))
            appointment_count = cursor.fetchone()[0]
            
            # Display stats
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Medical Records", record_count)
                st.metric("Appointments", appointment_count)
            
            with col_b:
                st.metric("Alzheimer's Analyses", analysis_count)
                if latest_analysis:
                    status_color = {
                        "Demented": "🔴",
                        "Nondemented": "🟢",
                        "Converted": "🟠"
                    }.get(latest_analysis[0], "⚪")
                    
                    st.metric(
                        "Latest Status", 
                        f"{status_color} {latest_analysis[0]}", 
                        f"Confidence: {latest_analysis[1]:.1%}"
                    )
            
            cursor.close()
            conn.close()
        
        # Schedule appointment
        st.subheader("Schedule Appointment")
        with st.form("schedule_appointment"):
            # Get list of doctors
            conn = get_db_connection()
            doctors = []
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT doctor_id, full_name FROM doctors ORDER BY full_name")
                doctors = cursor.fetchall()
                cursor.close()
                conn.close()
            
            # Form fields
            if doctors:
                doctor_options = {doc[1]: doc[0] for doc in doctors}
                selected_doctor = st.selectbox("Select Doctor", list(doctor_options.keys()))
                doctor_id = doctor_options[selected_doctor]
            else:
                doctor_id = st.session_state.get("user_id", 1)
                st.info("No other doctors available in system.")
            
            appt_date = st.date_input("Appointment Date", value=datetime.now().date() + timedelta(days=1))
            appt_time = st.time_input("Appointment Time", value=time(9, 0))
            appt_datetime = datetime.combine(appt_date, appt_time)
            reason = st.text_area("Reason for Visit")
            
            submit_appt = st.form_submit_button("Schedule Appointment")
            if submit_appt:
                if reason:
                    conn = get_db_connection()
                    if conn:
                        cursor = conn.cursor()
                        try:
                            cursor.execute("""
                                INSERT INTO appointments
                                (patient_id, doctor_id, appointment_date, reason, status)
                                VALUES (%s, %s, %s, %s, %s)
                            """, (patient_id, doctor_id, appt_datetime, reason, "Scheduled"))
                            
                            conn.commit()
                            st.success("✅ Appointment scheduled successfully.")
                            time_module.sleep(1)
                            st.rerun()
                        except mysql.connector.Error as e:
                            st.error(f"Error scheduling appointment: {e}")
                        finally:
                            cursor.close()
                            conn.close()
                else:
                    st.warning("Please provide a reason for the appointment.")
        
        # Show upcoming appointments
        st.subheader("Upcoming Appointments")
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT a.appointment_id, a.appointment_date, a.reason, a.status, d.full_name as doctor_name
                FROM appointments a
                JOIN doctors d ON a.doctor_id = d.doctor_id
                WHERE a.patient_id = %s AND a.appointment_date >= NOW()
                ORDER BY a.appointment_date ASC
            """, (patient_id,))
            
            appointments = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if appointments:
                for appt in appointments:
                    with st.expander(f"📅 {appt['appointment_date'].strftime('%Y-%m-%d %H:%M')} - {appt['status']}"):
                        st.write(f"**Doctor:** {appt['doctor_name']}")
                        st.write(f"**Reason:** {appt['reason']}")
                        st.write(f"**Status:** {appt['status']}")
                        
                        # Allow updating the status
                        new_status = st.selectbox(
                            "Update Status",
                            ["Scheduled", "Completed", "Cancelled", "No-show"],
                            index=["Scheduled", "Completed", "Cancelled", "No-show"].index(appt['status']),
                            key=f"status_{appt['appointment_id']}"
                        )
                        
                        if st.button("Update Status", key=f"update_{appt['appointment_id']}"):
                            conn = get_db_connection()
                            if conn:
                                cursor = conn.cursor()
                                try:
                                    cursor.execute("""
                                        UPDATE appointments
                                        SET status = %s
                                        WHERE appointment_id = %s
                                    """, (new_status, appt['appointment_id']))
                                    
                                    conn.commit()
                                    st.success("✅ Appointment status updated.")
                                    time_module.sleep(1)
                                    st.rerun()
                                except mysql.connector.Error as e:
                                    st.error(f"Error updating appointment status: {e}")
                                finally:
                                    cursor.close()
                                    conn.close()
            else:
                st.info("No upcoming appointments scheduled.")

# Display Alzheimer's analysis page
def display_alzheimer_analysis(patient_id, patient_info):
    st.header("🧠 Alzheimer's Disease Analysis")
    
    # Get existing features for the patient if available
    existing_features = get_patient_features(patient_id)
    
    # Create feature categories for better organization
    feature_categories = {
        "Cognitive Tests": ["CDRSB", "MMSE", "MOCA", "ADAS11", "ADAS13", "FAQ"],
        "Memory Tests": ["RAVLT_immediate", "RAVLT_learning", "RAVLT_forgetting", "RAVLT_perc_forgetting", "LDELTOTAL"],
        "Functional Tests": ["TRABSCOR", "DIGITSCOR", "mPACCdigit", "mPACCtrailsB"],
        "Patient Self-Report": ["EcogPtTotal", "EcogPtMem", "EcogPtLang", "EcogPtVisspat", "EcogPtPlan", "EcogPtOrgan", "EcogPtDivatt"],
        "Study Partner Report": ["EcogSPTotal", "EcogSPMem", "EcogSPLang", "EcogSPVisspat", "EcogSPPlan", "EcogSPOrgan", "EcogSPDivatt"],
        "Brain Measurements": ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain", "ICV", "FDG"],
        "Biomarkers": ["ABETA", "TAU", "PTAU", "APOE4"],
        "Demographics": ["AGE", "PTGENDER", "PTEDUCAT", "PTMARRY", "PTRACCAT", "PTETHCAT"],
        "Other Assessments": ["ADASQ4"]
    }
    
    # Get feature descriptions for tooltips
    feature_descriptions = get_feature_descriptions()
    
    # Historical data vs. new analysis
    tab1, tab2 = st.tabs(["New Analysis", "Analysis History"])
    
    with tab1:
        # Initialize input_data dictionary to store all features
        input_data = {}
        
        st.info("Enter values for as many features as possible. Hover over feature names for descriptions.")
        
        # Use tabs for feature categories to save space
        feature_tabs = st.tabs(list(feature_categories.keys()))
        
        for i, (category, features) in enumerate(feature_categories.items()):
            with feature_tabs[i]:
                cols = st.columns(2)
                for j, feature in enumerate(features):
                    col = cols[j % 2]
                    with col:
                        # Show tooltip with feature description
                        description = feature_descriptions.get(feature, "No description available")
                        st.markdown(f"**{feature}** ℹ️")
                        st.caption(description)
                        
                        # Use existing value as default if available
                        default_value = 0.0
                        if existing_features and feature in existing_features and existing_features[feature] is not None:
                            default_value = float(existing_features[feature])
                            
                        # Input field for feature
                        value = st.number_input(
                            f"{feature}", 
                            value=default_value,
                            step=0.01,
                            format="%.2f",
                            key=f"feature_{feature}"
                        )
                        input_data[feature] = value
        
        # Add any missing features with default values
        for feature in get_feature_columns():
            if feature not in input_data:
                input_data[feature] = 0.0
                
        # Make prediction
        col1, col2 = st.columns([3, 1])
        with col2:
            predict_button = st.button("🧠 Predict Alzheimer's Status", type="primary")
        with col1:
            save_only = st.button("💾 Save Features Only")
            
        if save_only:
            # Save feature values to database without prediction
            if store_features(patient_id, input_data):
                st.success("✅ Patient features saved to database")
                time_module.sleep(1)
                st.rerun()
            else:
                st.error("❌ Failed to save patient features")
        
        if predict_button:
            with st.spinner("Processing prediction..."):
                # Save feature values to database
                if store_features(patient_id, input_data):
                    st.success("✅ Patient features saved to database")
                    
                    # Run prediction
                    prediction, confidence = predict_alzheimer(input_data)
                    
                    # Store prediction results
                    analysis_id = store_prediction(patient_id, input_data, prediction, confidence)
                    
                    if analysis_id:
                        st.success("✅ Prediction results saved")
                        
                        # Create columns for results
                        st.markdown("### Prediction Results")
                        result_cols = st.columns([1, 1, 1])
                        
                        # Display prediction results
                        with result_cols[0]:
                            # Add appropriate icon and color based on prediction
                            if prediction == "Demented":
                                st.error("⚠️ Prediction: Demented")
                            elif prediction == "Nondemented":
                                st.success("✅ Prediction: Nondemented")
                            elif prediction == "Converted":
                                st.warning("⚠️ Prediction: Converted")
                            else:
                                st.info(f"🔍 Prediction: {prediction}")
                        
                        with result_cols[1]:
                            st.metric("Confidence", f"{confidence:.2%}")
                            
                        with result_cols[2]:
                            st.write("Prediction ID:", analysis_id)
                            st.write("Time:", datetime.now().strftime("%Y-%m-%d %H:%M"))
                        
                        # Display class probabilities as a bar chart if available
                        if hasattr(st.session_state, 'last_probabilities') and hasattr(st.session_state, 'last_prediction_classes'):
                            probs = st.session_state.last_probabilities
                            classes = st.session_state.last_prediction_classes
                            
                            prob_df = pd.DataFrame({
                                'Class': classes,
                                'Probability': probs
                            })
                            
                            st.markdown("### Probability Distribution")
                            fig, ax = plt.subplots(figsize=(2.5, 1.2))
                            colors = ['#ff9999' if c == 'Demented' else '#99ff99' if c == 'Nondemented' else '#ffcc99' for c in classes]
                            ax.bar(prob_df['Class'], prob_df['Probability'], color=colors)
                            ax.set_ylabel('Prob', fontsize=5)
                            ax.set_xlabel('Class', fontsize=5)
                            ax.set_ylim(0, 1)
                            ax.tick_params(axis='both', which='major', labelsize=4)
                            
                            for i, v in enumerate(probs):
                                ax.text(i, v + 0.01, f"{v:.2%}", ha='center')
                                
                            st.pyplot(fig)
                        
                        # Show feature importance
                        if clf is not None and hasattr(clf, 'feature_importances_'):
                            st.markdown("### Top Features Used in Prediction")
                            importance_plot = generate_feature_importance_plot()
                            if importance_plot:
                                st.image(importance_plot, use_column_width=True)
                        
                        # Generate report with AI assistant
                        st.markdown("### 🤖 AI-Generated Clinical Assessment")
                        with st.spinner("Generating clinical assessment..."):
                            # Prepare features for AI prompt
                            important_features = {
                                "MMSE": input_data["MMSE"],
                                "CDRSB": input_data["CDRSB"],
                                "ADAS13": input_data["ADAS13"],
                                "RAVLT_immediate": input_data["RAVLT_immediate"],
                                "Hippocampus": input_data["Hippocampus"],
                                "Entorhinal": input_data["Entorhinal"],
                                "AGE": input_data["AGE"],
                                "APOE4": input_data["APOE4"],
                                "TAU": input_data["TAU"],
                                "PTAU": input_data["PTAU"],
                                "ABETA": input_data["ABETA"]
                            }
                            
                            # Get recent medical records
                            records = get_patient_records(patient_id)
                            recent_records = records[:3] if records else []
                            
                            # Create prompt for AI
                            prompt = f"""
                            You are a clinical expert in Alzheimer's disease. Provide an assessment of the following patient based on the machine learning model prediction and clinical data.
                            
                            Patient Information:
                            - Name: {patient_info['full_name']}
                            - Age: {datetime.now().year - patient_info['birth_date'].year} years
                            - Gender: {patient_info['gender']}
                            
                            Model Prediction:
                            - Prediction: {prediction}
                            - Confidence: {confidence:.2%}
                            
                            Key Clinical Features:
                            """
                            
                            for feature, value in important_features.items():
                                desc = feature_descriptions.get(feature, "")
                                short_desc = desc.split("(")[0].strip() if "(" in desc else desc
                                prompt += f"- {feature}: {value} ({short_desc})\n"
                                
                            if recent_records:
                                prompt += "\nRecent Medical History:\n"
                                for record in recent_records:
                                    prompt += f"- Date: {record['visit_date']}, Diagnosis: {record['diagnosis']}\n"
                                    prompt += f"  Notes: {record['notes'][:100]}...\n" if len(record['notes']) > 100 else f"  Notes: {record['notes']}\n"
                            
                            prompt += """
                            Please provide:
                            1. A clinical assessment of the patient's Alzheimer's disease status based on the model prediction and clinical features
                            2. Interpretation of key biomarkers and test scores
                            3. Recommendations for further tests or interventions if appropriate
                            4. Brief list of relevant research citations supporting your assessment
                            
                            Format your response in clear sections with headings. Be concise but thorough.
                            """
                            
                            try:
                                response = model.generate_content(prompt)
                                assessment = response.text
                                
                                # Display the AI assessment
                                st.markdown(assessment)
                                
                                # Option to save assessment to medical records
                                if st.button("💾 Save Assessment to Medical Records"):
                                    diagnosis = f"AI-assisted Alzheimer's Analysis: {prediction}"
                                    if add_medical_record(patient_id, diagnosis, assessment):
                                        st.success("✅ Assessment saved to medical records")
                                    else:
                                        st.error("❌ Failed to save assessment")
                            except Exception as e:
                                st.error(f"Error generating AI assessment: {e}")
                    else:
                        st.error("❌ Failed to save prediction results")
                else:
                    st.error("❌ Failed to save patient features")
    
    # Analysis History tab
    with tab2:
        analyses = get_patient_analyses(patient_id)
        
        if not analyses:
            st.info("No previous analyses found for this patient.")
        else:
            # Convert to DataFrame for display
            df_analyses = pd.DataFrame(analyses)
            
            # Format dates
            df_analyses['analyzed_at'] = pd.to_datetime(df_analyses['analyzed_at'])
            df_analyses['Date'] = df_analyses['analyzed_at'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Prepare for display
            display_df = df_analyses[['analysis_id', 'prediction', 'confidence_score', 'Date']]
            display_df.columns = ['ID', 'Prediction', 'Confidence', 'Date']
            
            # Display analyses table
            st.dataframe(display_df, hide_index=True, use_container_width=True)
            
            # Create a visualization of prediction history
            st.subheader("Prediction History Visualization")
            
            # Prepare data for chart
            chart_df = df_analyses.copy()
            chart_df['confidence_score'] = chart_df['confidence_score'].astype(float)
            chart_df = chart_df.sort_values('analyzed_at')
            
            # Create color mapping for predictions
            color_map = {
                'Demented': '#ff9999',
                'Nondemented': '#99ff99',
                'Converted': '#ffcc99'
            }
            
            # Create plot with extremely small size
            fig, ax = plt.subplots(figsize=(3, 1.5))
            
            for pred in chart_df['prediction'].unique():
                pred_data = chart_df[chart_df['prediction'] == pred]
                ax.scatter(
                    pred_data['analyzed_at'], 
                    pred_data['confidence_score'],
                    label=pred,
                    color=color_map.get(pred, '#cccccc'),
                    s=100
                )
            
            # Connect points with lines
            ax.plot(chart_df['analyzed_at'], chart_df['confidence_score'], 'k--', alpha=0.3)
            
            # Format plot
            ax.set_ylabel('Confidence Score', fontsize=5)
            ax.set_xlabel('Date', fontsize=5)
            ax.set_title('Alzheimer\'s Prediction History', fontsize=6)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format y-axis as percentage
            import matplotlib.ticker as mtick
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            
            # Display plot
            st.pyplot(fig)
            
            # Option to compare analyses
            st.subheader("Compare Analyses")
            col1, col2 = st.columns(2)
            
            with col1:
                analysis_id1 = st.selectbox(
                    "Select First Analysis",
                    options=df_analyses['analysis_id'].tolist(),
                    format_func=lambda x: f"ID: {x} - {df_analyses[df_analyses['analysis_id']==x]['Date'].iloc[0]}"
                )
            
            with col2:
                remaining_ids = [id for id in df_analyses['analysis_id'].tolist() if id != analysis_id1]
                if remaining_ids:
                    analysis_id2 = st.selectbox(
                        "Select Second Analysis",
                        options=remaining_ids,
                        format_func=lambda x: f"ID: {x} - {df_analyses[df_analyses['analysis_id']==x]['Date'].iloc[0]}"
                    )
                    
                    if st.button("Compare Analyses"):
                        # Get the full data for both analyses
                        conn = get_db_connection()
                        if conn:
                            cursor = conn.cursor(dictionary=True)
                            try:
                                cursor.execute("""
                                    SELECT analysis_id, prediction, confidence_score, analyzed_at, input_features
                                    FROM alzheimers_analysis
                                    WHERE analysis_id IN (%s, %s)
                                """, (analysis_id1, analysis_id2))
                                
                                results = cursor.fetchall()
                                cursor.close()
                                conn.close()
                                
                                if len(results) == 2:
                                    # Extract data
                                    data1 = next(r for r in results if r['analysis_id'] == analysis_id1)
                                    data2 = next(r for r in results if r['analysis_id'] == analysis_id2)
                                    
                                    # Parse JSON features
                                    features1 = json.loads(data1['input_features'])
                                    features2 = json.loads(data2['input_features'])
                                    
                                    # Compare results
                                    st.markdown("### Analysis Comparison")
                                    
                                    # Display summary
                                    summary_cols = st.columns(2)
                                    with summary_cols[0]:
                                        st.markdown(f"**First Analysis** (ID: {analysis_id1})")
                                        st.markdown(f"Date: {data1['analyzed_at']}")
                                        st.markdown(f"Prediction: {data1['prediction']}")
                                        st.markdown(f"Confidence: {float(data1['confidence_score']):.2%}")
                                    
                                    with summary_cols[1]:
                                        st.markdown(f"**Second Analysis** (ID: {analysis_id2})")
                                        st.markdown(f"Date: {data2['analyzed_at']}")
                                        st.markdown(f"Prediction: {data2['prediction']}")
                                        st.markdown(f"Confidence: {float(data2['confidence_score']):.2%}")
                                    
                                    # Find key differences in features
                                    st.markdown("### Key Differences in Features")
                                    
                                    differences = []
                                    for feature in get_feature_columns():
                                        if feature in features1 and feature in features2:
                                            val1 = float(features1[feature])
                                            val2 = float(features2[feature])
                                            
                                            # Calculate absolute and percentage difference
                                            abs_diff = val2 - val1
                                            pct_diff = abs_diff / (val1 if val1 != 0 else 1) * 100
                                            
                                            # Consider significant changes
                                            if abs(pct_diff) > 10 or abs(abs_diff) > 0.5:
                                                differences.append({
                                                    'Feature': feature,
                                                    'First Value': val1,
                                                    'Second Value': val2,
                                                    'Absolute Change': abs_diff,
                                                    'Percentage Change': pct_diff
                                                })
                                    
                                    if differences:
                                        # Convert to DataFrame and sort by percentage change
                                        diff_df = pd.DataFrame(differences)
                                        diff_df = diff_df.sort_values('Percentage Change', key=abs, ascending=False)
                                        
                                        # Format for display
                                        display_diff = diff_df.copy()
                                        display_diff['Percentage Change'] = display_diff['Percentage Change'].apply(lambda x: f"{x:+.2f}%")
                                        display_diff['Absolute Change'] = display_diff['Absolute Change'].apply(lambda x: f"{x:+.2f}")
                                        
                                        # Display differences table
                                        st.dataframe(display_diff, hide_index=True, use_container_width=True)
                                    else:
                                        st.info("No significant differences found between the analyses.")
                            except Exception as e:
                                st.error(f"Error comparing analyses: {e}")
                else:
                    st.info("Need at least two analyses to compare.")

# Display medical records page
def display_medical_records(patient_id, patient_info):
    st.header("📋 Medical Records")
    
    # Get all medical records
    records = get_patient_records(patient_id)
    
    # Add new record form
    st.subheader("Add New Medical Record")
    with st.form("medical_record_form"):
        diagnosis = st.text_input("Diagnosis")
        visit_date = st.date_input("Visit Date", value=datetime.now().date())
        notes = st.text_area("Clinical Notes", height=150)
        
        submit = st.form_submit_button("Save Medical Record")
        
        if submit:
            if diagnosis and notes:
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    try:
                        cursor.execute("""
                            INSERT INTO medical_records 
                            (patient_id, diagnosis, visit_date, notes)
                            VALUES (%s, %s, %s, %s)
                        """, (patient_id, diagnosis, visit_date, notes))
                        
                        conn.commit()
                        st.success("✅ Medical record saved successfully")
                        
                        # Refresh the page after successful save
                        time_module.sleep(1)
                        st.rerun()
                    except mysql.connector.Error as e:
                        st.error(f"Error saving medical record: {e}")
                    finally:
                        cursor.close()
                        conn.close()
            else:
                st.warning("Please enter both diagnosis and notes.")
    
    # Display existing records
    st.subheader("Medical History")
    
    if not records:
        st.info("No medical records found for this patient.")
    else:
        # Organize records by year for better navigation
        records_df = pd.DataFrame(records)
        records_df['visit_date'] = pd.to_datetime(records_df['visit_date'])
        records_df['year'] = records_df['visit_date'].dt.year
        
        # Group by year
        years = sorted(records_df['year'].unique(), reverse=True)
        
        # Create tabs for years
        if len(years) > 1:
            year_tabs = st.tabs([str(year) for year in years])
            
            for i, year in enumerate(years):
                with year_tabs[i]:
                    year_records = records_df[records_df['year'] == year].sort_values('visit_date', ascending=False)
                    
                    for _, record in year_records.iterrows():
                        with st.expander(f"{record['visit_date'].strftime('%Y-%m-%d')} - {record['diagnosis']}"):
                            st.markdown(f"**Diagnosis:** {record['diagnosis']}")
                            st.markdown(f"**Date:** {record['visit_date'].strftime('%Y-%m-%d')}")
                            st.markdown(f"**Notes:**")
                            st.markdown(record['notes'])
                            
                            # Edit and delete options
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("🖊️ Edit", key=f"edit_{record['record_id']}"):
                                    st.session_state.edit_record_id = record['record_id']
                                    st.session_state.edit_diagnosis = record['diagnosis']
                                    st.session_state.edit_notes = record['notes']
                                    st.rerun()
                            
                            with col2:
                                if st.button("🗑️ Delete", key=f"delete_{record['record_id']}"):
                                    conn = get_db_connection()
                                    if conn:
                                        cursor = conn.cursor()
                                        try:
                                            cursor.execute("DELETE FROM medical_records WHERE record_id = %s", (record['record_id'],))
                                            conn.commit()
                                            st.success("Record deleted successfully")
                                            time_module.sleep(1)
                                            st.rerun()
                                        except mysql.connector.Error as e:
                                            st.error(f"Error deleting record: {e}")
                                        finally:
                                            cursor.close()
                                            conn.close()
        else:
            # If only one year, don't use tabs
            for _, record in records_df.sort_values('visit_date', ascending=False).iterrows():
                with st.expander(f"{record['visit_date'].strftime('%Y-%m-%d')} - {record['diagnosis']}"):
                    st.markdown(f"**Diagnosis:** {record['diagnosis']}")
                    st.markdown(f"**Date:** {record['visit_date'].strftime('%Y-%m-%d')}")
                    st.markdown(f"**Notes:**")
                    st.markdown(record['notes'])
                    
                    # Edit and delete options
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🖊️ Edit", key=f"edit_{record['record_id']}"):
                            st.session_state.edit_record_id = record['record_id']
                            st.session_state.edit_diagnosis = record['diagnosis']
                            st.session_state.edit_notes = record['notes']
                            st.rerun()
                    
                    with col2:
                        if st.button("🗑️ Delete", key=f"delete_{record['record_id']}"):
                            conn = get_db_connection()
                            if conn:
                                cursor = conn.cursor()
                                try:
                                    cursor.execute("DELETE FROM medical_records WHERE record_id = %s", (record['record_id'],))
                                    conn.commit()
                                    st.success("Record deleted successfully")
                                    time_module.sleep(1)
                                    st.rerun()
                                except mysql.connector.Error as e:
                                    st.error(f"Error deleting record: {e}")
                                finally:
                                    cursor.close()
                                    conn.close()
    
    # Handle record editing
    if hasattr(st.session_state, 'edit_record_id'):
        st.subheader("Edit Medical Record")
        
        with st.form("edit_record_form"):
            edit_diagnosis = st.text_input("Diagnosis", value=st.session_state.edit_diagnosis)
            edit_notes = st.text_area("Clinical Notes", value=st.session_state.edit_notes, height=150)
            
            col1, col2 = st.columns(2)
            with col1:
                update = st.form_submit_button("Update Record")
            
            with col2:
                cancel = st.form_submit_button("Cancel")
            
            if update:
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    try:
                        cursor.execute("""
                            UPDATE medical_records 
                            SET diagnosis = %s, notes = %s
                            WHERE record_id = %s
                        """, (edit_diagnosis, edit_notes, st.session_state.edit_record_id))
                        
                        conn.commit()
                        st.success("✅ Medical record updated successfully")
                        
                        # Clear edit state
                        del st.session_state.edit_record_id
                        del st.session_state.edit_diagnosis
                        del st.session_state.edit_notes
                        
                        # Refresh the page
                        time_module.sleep(1)
                        st.rerun()
                    except mysql.connector.Error as e:
                        st.error(f"Error updating medical record: {e}")
                    finally:
                        cursor.close()
                        conn.close()
            
            if cancel:
                # Clear edit state
                del st.session_state.edit_record_id
                del st.session_state.edit_diagnosis
                del st.session_state.edit_notes
                st.rerun()

# Display AI assistant page
def display_ai_assistant(patient_id, patient_info, doctor_id):
    st.header("💬 AI Clinical Assistant")
    
    # Sidebar for context options
    with st.sidebar:
        st.subheader("Assistant Context")
        include_features = st.toggle("Include Clinical Features", value=True)
        include_records = st.toggle("Include Medical Records", value=True)
        include_analyses = st.toggle("Include Analysis History", value=True)
        
        # Reset conversation button
        if st.button("🔄 Reset Conversation"):
            # Clear the conversation in session state
            st.session_state.chat_history = []
            
            # Optional: Delete chat history from database if you want to completely remove it
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("DELETE FROM chat_logs WHERE patient_id = %s AND doctor_id = %s", 
                                   (patient_id, doctor_id))
                    conn.commit()
                    st.success("Chat history cleared from database")
                except mysql.connector.Error as e:
                    st.error(f"Error clearing chat history: {e}")
                finally:
                    cursor.close()
                    conn.close()
                    
            st.rerun()
    
    # Always load chat history from database when displaying the assistant page
    # This ensures we have the most recent history, even if coming from another section
    chat_history = get_chat_history(patient_id, doctor_id)
    
    # Update the session state with the loaded history
    st.session_state.chat_history = chat_history
    
    # Display chat messages in a container with scrolling
    chat_container = st.container(height=400, border=True)
    
    with chat_container:
        for i, (sender, msg) in enumerate(st.session_state.chat_history):
            message(msg, is_user=(sender == "You"), key=f"msg_{i}")
    
    # Add scroll to bottom button
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("⬇️ Scroll to Bottom"):
            st.components.v1.html(
                """
                <script>
                    function scrollToBottom() {
                        const containers = document.getElementsByClassName('stChatFloatingInputContainer');
                        if (containers.length > 0) {
                            const chatContainer = containers[0].parentElement;
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        }
                    }
                    setTimeout(scrollToBottom, 100);
                </script>
                """,
                height=0
            )
    
    # Chat input
    user_input = st.chat_input("Ask about this patient or how to interpret results...")
    
    # Auto-scroll to bottom when new messages are added
    if st.session_state.chat_history:
        st.components.v1.html(
            """
            <script>
                function scrollToBottom() {
                    const containers = document.getElementsByClassName('stChatFloatingInputContainer');
                    if (containers.length > 0) {
                        const chatContainer = containers[0].parentElement;
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }
                }
                setTimeout(scrollToBottom, 100);
            </script>
            """,
            height=0
        )
    
    # Get patient data for context when needed
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append(("You", user_input))
        
        # Save user message to database
        save_chat_message(patient_id, doctor_id, user_input, "Doctor")
        
        # Prepare context for AI
        context = f"""
        Patient Information:
        - Name: {patient_info['full_name']}
        - Age: {datetime.now().year - patient_info['birth_date'].year} years
        - Gender: {patient_info['gender']}
        - ID: {patient_id}
        """
        
        # Add clinical features if requested
        if include_features:
            features = get_patient_features(patient_id)
            if features:
                # Filter to most important features for context
                important_features = [
                    "MMSE", "CDRSB", "ADAS13", "RAVLT_immediate", 
                    "Hippocampus", "AGE", "APOE4", "TAU", "ABETA"
                ]
                
                feature_context = "\nKey Clinical Features:\n"
                feature_descriptions = get_feature_descriptions()
                
                for feature in important_features:
                    if feature in features and features[feature] is not None:
                        desc = feature_descriptions.get(feature, "")
                        short_desc = desc.split("(")[0].strip() if "(" in desc else desc
                        feature_context += f"- {feature}: {features[feature]} ({short_desc})\n"
                
                context += feature_context
        
        # Add medical history if requested
        if include_records:
            records = get_patient_records(patient_id)
            if records:
                recent_records = records[:3]  # Get 3 most recent
                
                records_context = "\nRecent Medical History:\n"
                for record in recent_records:
                    records_context += f"- Date: {record['visit_date']}, Diagnosis: {record['diagnosis']}\n"
                    summary = record['notes'][:100] + "..." if len(record['notes']) > 100 else record['notes']
                    records_context += f"  Notes: {summary}\n"
                
                context += records_context
        
        # Add analysis history if requested
        if include_analyses:
            analyses = get_patient_analyses(patient_id)
            if analyses:
                recent_analyses = analyses[:2]  # Get 2 most recent
                
                analyses_context = "\nRecent Alzheimer's Analyses:\n"
                for analysis in recent_analyses:
                    analyses_context += f"- Date: {analysis['analyzed_at']}, Prediction: {analysis['prediction']}, Confidence: {float(analysis['confidence_score']):.1%}\n"
                
                context += analyses_context
        
        # Build conversation history
        conversation = "\n".join([f"{'User' if sender == 'You' else 'Assistant'}: {msg}" for sender, msg in st.session_state.chat_history])
        
        # Create complete prompt
        prompt = f"""
        You are a clinical AI assistant specializing in Alzheimer's disease assessment and interpretation.
        You help doctors understand patient data, interpret test results, and provide evidence-based insights.
        
        Here is the current patient context:
                {context}
                
        Conversation history:
                {conversation}
                
        When responding, please:
        1. Be concise but thorough
        2. Cite relevant research when appropriate
        3. Avoid making definitive diagnoses, but help interpret the data
        4. Format your response with clear sections when needed
        5. If you refer to research studies or clinical guidelines, provide brief citations
        6. If you don't know something, be honest about limitations
        
        Your response should be directly helpful to a doctor treating this patient for potential Alzheimer's disease.
        """
        
        with st.spinner("Generating response..."):
            try:
                response = model.generate_content(prompt)
                assistant_message = response.text
                
                # Add to chat history and display
                st.session_state.chat_history.append(("Assistant", assistant_message))
                
                # Save to database
                save_chat_message(patient_id, doctor_id, assistant_message, "Assistant")
                
                # Rerun to display the new message
                st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {e}")
        
    # Display helpful shortcuts
    with st.expander("💡 Helpful Questions to Ask"):
        st.markdown("""
        - What do the MMSE and CDRSB scores indicate for this patient?
        - How do I interpret the Hippocampus volume in relation to Alzheimer's?
        - What treatment options are recommended for a patient with these biomarkers?
        - Explain the significance of these APOE4 results.
        - What additional tests would you recommend for this patient?
        - Summarize the key findings from the patient's history and test results.
        - What is the clinical significance of the tau/amyloid ratio?
        - What lifestyle modifications might help this patient?
        """)

# Display analytics page
def display_analytics(patient_id, patient_info):
    st.header("📊 Analytics Dashboard")
    
    # Get patient analyses
    analyses = get_patient_analyses(patient_id)
    
    if not analyses:
        st.info("No analysis data available for this patient yet.")
        return
    
    # Convert to DataFrame
    df_analyses = pd.DataFrame(analyses)
    df_analyses['confidence_score'] = df_analyses['confidence_score'].astype(float)
    df_analyses['analyzed_at'] = pd.to_datetime(df_analyses['analyzed_at'])
    
    # Summary statistics
    st.subheader("Patient Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_analyses = len(df_analyses)
        st.metric("Total Analyses", total_analyses)
    
    with col2:
        latest_analysis = df_analyses.iloc[0]
        status_emoji = {
            "Demented": "🔴",
            "Nondemented": "🟢",
            "Converted": "🟠"
        }.get(latest_analysis['prediction'], "⚪")
        
        st.metric(
            "Current Status", 
            f"{status_emoji} {latest_analysis['prediction']}", 
            f"{latest_analysis['confidence_score']:.1%}"
        )
    
    with col3:
        time_span = None
        if len(df_analyses) > 1:
            first_date = df_analyses['analyzed_at'].min()
            last_date = df_analyses['analyzed_at'].max()
            days = (last_date - first_date).days
            if days > 365:
                time_span = f"{days/365:.1f} years"
            else:
                time_span = f"{days} days"
        
        st.metric("Monitoring Period", time_span or "N/A")
    
    # Trend analysis
    if len(df_analyses) > 1:
        st.subheader("Disease Progression Trend")
        
        # Sort by date for trend analysis
        trend_df = df_analyses.sort_values('analyzed_at')
        
        # Create line chart of prediction confidence over time
        fig, ax = plt.subplots(figsize=(3, 1.5))
        
        # Plot each prediction type with different color
        for pred in trend_df['prediction'].unique():
            pred_data = trend_df[trend_df['prediction'] == pred]
            color = {
                'Demented': 'red',
                'Nondemented': 'green',
                'Converted': 'orange'
            }.get(pred, 'blue')
            
            ax.plot(
                pred_data['analyzed_at'], 
                pred_data['confidence_score'],
                'o-',
                label=pred,
                color=color
            )
        
        # Format plot
        ax.set_xlabel('Date', fontsize=5)
        ax.set_ylabel('Confidence', fontsize=5)
        ax.set_title('Progression', fontsize=6)
        ax.legend(fontsize=4)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        import matplotlib.ticker as mtick
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        # Display plot
        st.pyplot(fig)
        
        # Add prediction distribution
        st.subheader("Prediction Distribution")
        
        # Count predictions
        prediction_counts = df_analyses['prediction'].value_counts()
        
        # Create pie chart with extremely small size
        fig, ax = plt.subplots(figsize=(2, 2))
        
        ax.pie(
            prediction_counts, 
            labels=prediction_counts.index,
            autopct='%1.1f%%',
            colors=['#ff9999', '#99ff99', '#ffcc99', '#cccccc'],
            startangle=90,
            explode=[0.05] * len(prediction_counts)
        )
        
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Display pie chart
        st.pyplot(fig)
    
    # Feature trend analysis if available
    st.subheader("Feature Trends")
    
    conn = get_db_connection()
    if conn:
        feature_data = []
        cursor = conn.cursor(dictionary=True)
        try:
            # Get feature data from analyses
            for analysis in analyses:
                cursor.execute("""
                    SELECT input_features, analyzed_at
                    FROM alzheimers_analysis
                    WHERE analysis_id = %s
                """, (analysis['analysis_id'],))
                
                result = cursor.fetchone()
                if result and result['input_features']:
                    try:
                        features = json.loads(result['input_features'])
                        features['analyzed_at'] = result['analyzed_at']
                        feature_data.append(features)
                    except json.JSONDecodeError:
                        continue
        except mysql.connector.Error as e:
            st.error(f"Error fetching feature data: {e}")
        finally:
            cursor.close()
            conn.close()
        
        if feature_data:
            # Convert to DataFrame
            features_df = pd.DataFrame(feature_data)
            
            # Select important features to track
            important_features = [
                "MMSE", "CDRSB", "ADAS13", "RAVLT_immediate", 
                "Hippocampus", "Entorhinal", "ABETA", "TAU"
            ]
            
            # Allow user to select features to visualize
            selected_features = st.multiselect(
                "Select features to visualize",
                options=important_features,
                default=important_features[:3]
            )
            
            if selected_features:
                # Create line chart with extremely small size
                fig, ax = plt.subplots(figsize=(3, 1.5))
                
                # Get feature descriptions
                feature_descriptions = get_feature_descriptions()
                
                # Plot each selected feature
                for feature in selected_features:
                    if feature in features_df.columns:
                        ax.plot(
                            features_df['analyzed_at'], 
                            features_df[feature].astype(float),
                            'o-',
                            label=feature
                        )
                
                # Format plot
                ax.set_xlabel('Date', fontsize=5)
                ax.set_ylabel('Value', fontsize=5)
                ax.set_title('Clinical Feature Trends Over Time', fontsize=6)
                ax.legend(fontsize=4)
                ax.grid(True, alpha=0.3)
                
                # Display plot
                st.pyplot(fig)
                
                # Display feature descriptions for reference
                st.subheader("Feature Descriptions")
                for feature in selected_features:
                    desc = feature_descriptions.get(feature, "No description available")
                    st.markdown(f"**{feature}**: {desc}")
            else:
                st.info("Please select at least one feature to visualize.")
        else:
            st.info("No feature data available for trend analysis.")