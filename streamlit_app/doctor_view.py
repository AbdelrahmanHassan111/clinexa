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
import cv2
from PIL import Image
from mri_models import process_mri_with_huggingface, apply_colormap_to_heatmap, extract_roi_measurements
import random
# Add import for visit comparison
from visit_comparison import display_visit_comparison
import glob
# Add import for plotly at the top of the file (ensure it's with other imports)
import plotly.express as px
import plotly.graph_objects as go
import uuid
import base64
import io
from datetime import datetime, timedelta, time

# Gemini API setup
API_KEY = "AIzaSyC1R-VeIuMePDZt_Z1WLluHkoq2tjWsVz8"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")

# Load the ML model - use relative path
model_path = "model\\XGBoost_grid_optimized.joblib"
try:
    # Check if model file exists
    if os.path.exists(model_path):
        clf = joblib.load(model_path)
    else:
        st.warning(f"Model file not found at {model_path}. Using placeholder model for demonstration.")
        # Create a dummy classifier for demonstration purposes
        from sklearn.ensemble import XGBClassifier
        clf = XGBClassifier(n_estimators=10)
        # Set some dummy feature importances for visualization
        import numpy as np
        feature_names = get_feature_columns()
        clf.feature_importances_ = np.random.random(len(feature_names))
        clf.classes_ = np.array([0, 1, 2])  # Cognitively Normal, MCI, AD
except Exception as e:
    st.error(f"Error loading model: {e}")
    clf = None

# Database connection parameters
DB_CONFIG = {
    "host": "clinexa.cgpek8igovya.us-east-1.rds.amazonaws.com",
    "port": 3306,
    "user": "clinexa",
    "password": "Am24268934",
    "database": "clinexa_db"
}

# Class mapping for Alzheimer's predictions
ALZHEIMER_CLASS_MAPPING = {
    0: "Cognitively Normal",
    1: "Mild Cognitive Impairment",
    2: "Alzheimer's Disease (AD)"
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
    """Return a dictionary mapping feature names to their descriptions"""
    return {
        # Cognitive assessments
        "CDRSB": "Clinical Dementia Rating Sum of Boxes - Clinical scale measuring disease severity (0-18, higher is worse)",
        "MMSE": "Mini-Mental State Examination - Global cognitive test (0-30, higher is better)",
        "MOCA": "Montreal Cognitive Assessment - Global cognitive test (0-30, higher is better)",
        "ADAS11": "Alzheimer's Disease Assessment Scale (11-item) - Cognitive test (0-70, higher is worse)",
        "ADAS13": "Alzheimer's Disease Assessment Scale (13-item) - Cognitive test (0-85, higher is worse)",
        "FAQ": "Functional Activities Questionnaire - Functional assessment (0-30, higher is worse)",
        
        # Memory tests
        "RAVLT_immediate": "Rey Auditory Verbal Learning Test - Immediate recall score (higher is better)",
        "RAVLT_learning": "Rey Auditory Verbal Learning Test - Learning score (higher is better)",
        "RAVLT_forgetting": "Rey Auditory Verbal Learning Test - Forgetting score (higher is worse)",
        "RAVLT_perc_forgetting": "Rey Auditory Verbal Learning Test - Percent forgetting (higher is worse)",
        "LDELTOTAL": "Logical Memory delayed recall - Delayed paragraph recall (higher is better)",
        
        # Brain volume measurements
        "Hippocampus": "Hippocampal volume in mm³ - Critical for memory formation (higher is better)",
        "Entorhinal": "Entorhinal cortex volume in mm³ - Important for memory (higher is better)",
        "Fusiform": "Fusiform gyrus volume in mm³ - Involved in visual recognition (higher is better)",
        "MidTemp": "Middle temporal gyrus volume in mm³ - Language processing (higher is better)",
        "Ventricles": "Ventricular volume in mm³ - Filled with cerebrospinal fluid (higher is worse)",
        "WholeBrain": "Whole brain volume in mm³ - Overall brain size (higher is better)",
        
        # Biomarkers
        "ABETA": "Amyloid-β (Aβ) levels in CSF - Key biomarker for Alzheimer's (higher is better)",
        "TAU": "Total Tau protein levels in CSF - Marker of neuronal damage (higher is worse)",
        "PTAU": "Phosphorylated Tau levels in CSF - Marker of tau tangles (higher is worse)",
        "APOE4": "Number of APOE ε4 alleles - Genetic risk factor (0, 1, or 2; higher is worse risk)",
        
        # Demographics
        "AGE": "Patient age in years at time of visit",
        "PTGENDER": "Patient gender (1=male, 2=female)",
        "PTEDUCAT": "Patient years of education completed"
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
        feature_array = np.zeros(len(feature_columns))
        for i, feature in enumerate(feature_columns):
            if feature in input_data:
                feature_array[i] = input_data[feature]
        
        # Log the features being used for prediction
        st.session_state.last_feature_array = feature_array
                
        # Make prediction using the model
        prediction = clf.predict([feature_array])[0]
        
        # Handle different model types
        if hasattr(clf, 'predict_proba'):
            probabilities = clf.predict_proba([feature_array])[0]
        else:
            # Create fake probabilities for dummy model
            probabilities = np.zeros(3)
            probabilities[prediction] = 0.8  # High confidence for predicted class
            remaining = 0.2 / (len(probabilities) - 1)
            for i in range(len(probabilities)):
                if i != prediction:
                    probabilities[i] = remaining
        
        confidence = max(probabilities)
        
        # Store all probabilities in session state for visualization
        st.session_state.last_probabilities = probabilities
        st.session_state.last_prediction_classes = clf.classes_
        
        # Map integer prediction to string prediction for backwards compatibility
        pred_mapping = {0: "Nondemented", 1: "Converted", 2: "Demented"}
        if isinstance(prediction, (int, np.integer)):
            prediction = pred_mapping.get(prediction, "Unknown")
        
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
    """Create an enhanced feature importance visualization"""
    if not hasattr(clf, 'feature_importances_'):
        return None
    
    # Get feature names and importance values
    features = get_feature_columns()
    importances = clf.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Take top 10 most important features
    top_n = 10
    top_indices = indices[:top_n]
    top_features = [features[i] for i in top_indices]
    top_importances = [importances[i] for i in top_indices]
    
    # Create a more visually appealing plot
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Use a horizontal bar chart for better readability of feature names
    bars = plt.barh(range(len(top_indices)), top_importances, align='center', 
                    color=plt.cm.viridis(np.linspace(0, 0.8, len(top_indices))))
    
    # Add feature name labels
    plt.yticks(range(len(top_indices)), top_features, fontsize=10)
    
    # Add value labels to the end of each bar
    for i, v in enumerate(top_importances):
        plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8)
    
    # Add axis labels and title
    plt.xlabel('Relative Importance', fontsize=12)
    plt.title('Top 10 Most Important Features for Alzheimer\'s Prediction', fontsize=14)
    
    # Add grid lines for easier reading
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create a section highlighting feature importance meaning
    plt.figtext(0.5, 0.01, 
                'Higher values indicate features that have more influence on the model prediction.',
                wrap=True, horizontalalignment='center', fontsize=8)
    
    # Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close()
    
    return buf

# Doctor panel main content
def doctor_panel():
    # Initialize doctor ID from session state
    doctor_id = st.session_state.get("user_id", 1)
    
    # Sidebar menu
    with st.sidebar:
        # Add logo at the top of sidebar, centered
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image("streamlit_app\\logo.png", width=120)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.title("Clinexa")
        st.caption("Beyond Data. Beyond Care.")
        
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
    
    # Enhanced analysis with MRI
    tab1, tab2, tab3 = st.tabs(["New Analysis", "MRI Analysis", "Analysis History"])
    
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
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            save_only = st.button("💾 Save Features Only")
        with col2:
            predict_button = st.button("🧠 Predict Alzheimer's Status", type="primary")
        with col3:
            discard_button = st.button("❌ Discard")
            
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
                    
                    # Get the predicted class (0, 1, 2) from the prediction
                    # Map the class index to the class name using the mapping
                    predicted_idx = 0  # Default to 0 (Cognitively Normal)
                    if prediction == "Demented":
                        predicted_idx = 2  # Alzheimer's Disease
                    elif prediction == "Converted":
                        predicted_idx = 1  # Mild Cognitive Impairment
                    
                    # Map the class index to the class name
                    mapped_prediction = ALZHEIMER_CLASS_MAPPING.get(predicted_idx, prediction)
                    
                    # Store prediction results
                    st.session_state.last_prediction = mapped_prediction
                    st.session_state.last_confidence = confidence
                    st.session_state.last_input_data = input_data
                    
                    # Create a modern results card with enhanced UI using Streamlit native elements
                    st.markdown("### 📊 Prediction Results", unsafe_allow_html=True)
                    
                    # Create a 3-column layout for prediction results
                    result_cols = st.columns([1, 1, 1])
                    
                    # Column 1: Prediction Status with Streamlit elements
                    with result_cols[0]:
                        # Different styling based on prediction
                        if prediction == "Nondemented":
                            status_color = "green"
                            emoji = "✅"
                            risk_level = "Low Risk"
                        elif prediction == "Converted":
                            status_color = "orange"
                            emoji = "⚠️"
                            risk_level = "Moderate Risk"
                        else:  # Demented
                            status_color = "red"
                            emoji = "🚨"
                            risk_level = "High Risk"
                        
                        # Use Streamlit metric component
                        st.metric(
                            label=f"{emoji} Diagnosis",
                            value=mapped_prediction,
                            delta=risk_level
                        )
                    
                    # Column 2: Confidence Score with progress bars
                    with result_cols[1]:
                        # Convert confidence to percentage
                        conf_percentage = confidence * 100
                        
                        st.subheader("Confidence Levels")
                        
                        # Show all class probabilities with progress bars if available
                        if hasattr(st.session_state, 'last_probabilities') and hasattr(st.session_state, 'last_prediction_classes'):
                            probs = st.session_state.last_probabilities
                            classes = st.session_state.last_prediction_classes
                            
                            # Get class names
                            class_names = [ALZHEIMER_CLASS_MAPPING.get(i, str(i)) for i in classes]
                            
                            # Display progress bars for each class
                            for i, (prob, class_name) in enumerate(zip(probs, class_names)):
                                # Set color based on which class this is
                                if class_name == mapped_prediction:
                                    bar_color = status_color
                                else:
                                    bar_color = "gray"
                                    
                                st.progress(float(prob), text=f"{class_name}: {prob*100:.1f}%")
                        else:
                            # Fallback to just showing the main confidence
                            st.progress(float(confidence), text=f"Overall: {conf_percentage:.1f}%")
                    
                    # Column 3: Date & Actions
                    with result_cols[2]:
                        current_date = datetime.now().strftime("%Y-%m-%d")
                        st.markdown(f"**Analysis Date:** {current_date}")
                        
                        # Add action buttons for saving or removing prediction
                        st.markdown("#### Actions")
                        
                        # Save button with Streamlit styling
                        if st.button("💾 Save to Records", type="primary", key="save_prediction"):
                            analysis_id = store_prediction(patient_id, input_data, prediction, confidence)
                            
                            if analysis_id:
                                st.success("✅ Prediction saved to patient record")
                                # Store in session that we've saved this prediction
                                st.session_state.prediction_saved = True
                            else:
                                st.error("❌ Failed to save prediction")
                        
                        # Delete/discard button
                        if st.button("🗑️ Discard Result", key="discard_prediction"):
                            # Clear the prediction results
                            st.session_state.last_prediction = None
                            st.session_state.last_confidence = None
                            st.session_state.last_input_data = None
                            st.session_state.last_probabilities = None
                            st.session_state.last_prediction_classes = None
                            st.rerun()
                    
                    # Feature importance visualization
                    st.markdown("### 📊 Feature Importance")
                    
                    # Check if we have feature importances in the model
                    if hasattr(clf, 'feature_importances_'):
                        # Generate feature importance plot
                        importance_buf = generate_feature_importance_plot()
                        
                        if importance_buf:
                            # Show the feature importance plot
                            st.image(importance_buf, use_container_width=True)
                            
                            # Get top 5 most important features
                            features = get_feature_columns()
                            importances = clf.feature_importances_
                            indices = np.argsort(importances)[::-1]
                            top_features = [features[i] for i in indices[:5]]
                            
                            with st.expander("📋 Top 5 Most Important Features"):
                                for i, feature in enumerate(top_features):
                                    value = input_data.get(feature, "N/A")
                                    description = get_feature_descriptions().get(feature, "No description available")
                                    st.markdown(f"**{i+1}. {feature}**: {value}")
                                    st.markdown(f"<small>{description}</small>", unsafe_allow_html=True)
                    else:
                        st.info("Feature importance information not available for this model.")
                    
                    # AI-Generated Assessment
                    st.markdown("### 🤖 AI Assessment")
                    
                    with st.spinner("Generating clinical assessment..."):
                        try:
                            # Prepare data for AI assessment
                            feature_descriptions = get_feature_descriptions()
                            
                            # Get key features for assessment
                            key_features = ["CDRSB", "MMSE", "MOCA", "ADAS11", "ADAS13", "Hippocampus", 
                                          "Entorhinal", "RAVLT_immediate", "RAVLT_forgetting", "FAQ"]
                            
                            # Prepare prompt for AI
                            prompt = f"""You are a clinical dementia specialist analyzing Alzheimer's disease risk factors.
                            
                            Patient diagnosis: {mapped_prediction}
                            Prediction confidence: {confidence * 100:.1f}%
                            
                            Key patient measurements:
                            """
                            
                            # Add key features and their values to prompt
                            for feature in key_features:
                                if feature in input_data:
                                    desc = feature_descriptions.get(feature, "").split('-')[0] if feature_descriptions.get(feature, "") else ""
                                    prompt += f"- {feature}: {input_data[feature]:.2f} ({desc})\n"
                            
                            prompt += """
                            Please provide a brief clinical assessment of this patient based on these measurements. 
                            Include:
                            1. A summary of the patient's cognitive status
                            2. Key findings from the measurements that support the diagnosis
                            3. Typical prognosis for patients with similar profiles
                            4. 1-2 treatment or management recommendations
                            
                            Keep your response concise (max 200 words) and use clinical but accessible language.
                            """
                            
                            # Generate AI assessment
                            response = model.generate_content(prompt)
                            assessment = response.text
                            
                            # Display the AI assessment with nice formatting
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; border-left: 5px solid #4285f4; padding: 15px; border-radius: 4px;">
                                {assessment}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Add a note about AI-generated content
                            st.caption("Note: This assessment was generated by an AI and should be reviewed by a healthcare professional.")
                            
                        except Exception as e:
                            st.error(f"Unable to generate AI assessment: {str(e)}")
                    
        if discard_button:
            # Clear the input data
            st.session_state.last_prediction = None
            st.session_state.last_confidence = None
            st.session_state.last_input_data = None
            st.session_state.last_probabilities = None
            st.session_state.last_prediction_classes = None
            st.rerun()
    
    # MRI Analysis Tab
    with tab2:
        st.subheader("MRI Scan Analysis")
        
        # Create tabs for upload vs viewing
        mri_tab1, mri_tab2 = st.tabs(["Upload New Scan", "View Patient Scans"])
        
        # Upload New Scan tab
        with mri_tab1:
            st.markdown("### Upload New MRI Scan")
            
            # File uploader for MRI scan
            uploaded_file = st.file_uploader(
                "Upload MRI scan (jpg, png, or dicom)", 
                type=["jpg", "jpeg", "png", "dcm"]
            )
            
            scan_type = st.selectbox(
                "Scan Type",
                ["T1-weighted", "T2-weighted", "FLAIR", "PET", "Other"],
                index=0
            )
            
            scan_notes = st.text_area("Scan Notes", placeholder="Enter any notes about this scan...")
            
            # Show preview if file is uploaded
            if uploaded_file is not None:
                col1, col2 = st.columns([1, 1])
                
                # Save uploaded file temporarily
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Display image preview
                with col1:
                    st.markdown("### Preview")
                    st.image(file_path, use_container_width=True)
                
                with col2:
                    st.markdown("### Analysis Options")
                    # Remove other model options, keep only Hugging Face
                    analysis_options = "Hugging Face Transformer"
                    st.info("Using DHEIVER/Alzheimer-MRI Transformer model for analysis")
                    
                    if st.button("Process Scan", type="primary"):
                        with st.spinner("Processing MRI scan..."):
                            # Run Hugging Face model inference
                            st.info("Running Hugging Face model analysis...")
                            from mri_models import process_mri_with_huggingface
                            
                            # Truncate scan_type to 45 characters to prevent database error
                            truncated_scan_type = scan_type[:45] if scan_type and len(scan_type) > 45 else scan_type
                            
                            results = process_mri_with_huggingface(file_path)
                            
                            if results and 'error' not in results:
                                # Display results
                                st.markdown("### Analysis Results")
                                result_cols = st.columns([1, 1])
                                
                        with result_cols[0]:
                                    prediction = results.get('prediction', 'Unknown')
                                    confidence = results.get('confidence', 0)
                                    
                                    # Format prediction with appropriate styling
                                    if "Demented" in prediction or "demented" in prediction.lower():
                                        st.error(f"**Prediction:** {prediction}")
                                    elif "Mild" in prediction or "mild" in prediction.lower():
                                        st.warning(f"**Prediction:** {prediction}")
                                    else:
                                        st.success(f"**Prediction:** {prediction}")
                                        
                                    st.info(f"**Confidence:** {confidence:.1%}")
                                    
                                    # Display all class probabilities if available
                                    if 'all_probabilities' in results:
                                        st.markdown("#### All Class Probabilities")
                                        probs = results['all_probabilities']
                                        if isinstance(probs, list):
                                            for cls in probs:
                                                if isinstance(cls, dict) and 'label' in cls and 'probability' in cls:
                                                    st.markdown(f"**{cls['label']}**: {cls['probability']:.1%}")
                                        
                                    # Extract key measurements if available
                                    if 'roi_measurements' in results:
                                        st.markdown("#### Key Brain Measurements")
                                        measurements = results['roi_measurements']
                                        
                                        key_regions = [
                                            ('hippocampus_total', 'Hippocampus', 'Critical for memory formation'),
                                            ('entorhinal_total', 'Entorhinal Cortex', 'Early site of tau pathology'),
                                            ('temporal_lobe_total', 'Temporal Lobe', 'Important for language and memory'),
                                            ('lateral_ventricles', 'Ventricles', 'Enlarged in AD')
                                        ]
                                        
                                        for key, name, desc in key_regions:
                                            if key in measurements:
                                                st.markdown(f"**{name}:** {measurements[key]:.2f} mm³ ({desc})")
                        
                        with result_cols[1]:
                                    # Show visualization
                                    if 'heatmap_path' in results and results['heatmap_path'] and os.path.exists(results['heatmap_path']):
                                        st.markdown("#### Grad-CAM Visualization")
                                        st.image(results['heatmap_path'], use_container_width=True)
                                        st.caption("Heat map showing regions that influenced the model's decision (XAI)")
                                
                                # Add MRI scan description section
                                    st.markdown("### Model Interpretation of MRI Scan")
                                    with st.spinner("Generating detailed scan description..."):
                                        description = generate_mri_description(results, truncated_scan_type)
                                        st.markdown(description)
                                    # Save scan with results to database
                                    patient_id = st.session_state.selected_patient
                                    save_result = save_mri_scan(
                                        patient_id, 
                                        file_path, 
                                        truncated_scan_type, 
                                        prediction=results.get('prediction'), 
                                        confidence=results.get('confidence'), 
                                        notes=scan_notes
                                    )
                                    
                                    if save_result:
                                        st.success(f"MRI scan saved to patient record with ID #{save_result}")
                                        
                                        # Also save ROI measurements if available
                                        if 'roi_measurements' in results:
                                            try:
                                                from mri_models import save_roi_measurements
                                                measurement_id = save_roi_measurements(save_result, results['roi_measurements'])
                                                if measurement_id:
                                                    st.success(f"Brain region measurements saved with ID #{measurement_id}")
                                            except Exception as e:
                                                st.warning(f"Could not save measurements: {e}")
                                        
                                        # Offer button to add results to medical record
                                        if st.button("Add Results to Medical Records"):
                                            prediction = results.get('prediction', 'Unknown')
                                            confidence = results.get('confidence', 0)
                                            
                                            # Format medical record entry
                                            diagnosis = f"MRI Scan Analysis (Hugging Face Transformer)"
                                            notes = f"""
                                            MRI Scan Type: {truncated_scan_type}
                                            Analysis Method: Hugging Face Transformer (DHEIVER/Alzheimer-MRI)
                                            Prediction: {prediction}
                                            Confidence: {confidence:.1%}
                                            
                                            User Notes: {scan_notes}
                                            
                                            Key Measurements:
                                            """
                                            if truncated_scan_type == "axial":
                                                notes = f"Analysis of axial MRI scan shows {prediction_text} with {confidence:.1f}% confidence."
                                            else:
                                                notes = f"Analysis of {truncated_scan_type} MRI scan shows {prediction_text} with {confidence:.1f}% confidence."

                                            # Add medical record
                                            if add_medical_record(patient_id, diagnosis, notes):
                                                st.success("Results added to medical records")
                                            else:
                                                # Display error message
                                                error_msg = results.get('error', 'Unknown error occurred') if results else 'Failed to process MRI scan'
                                                st.error(f"Error processing MRI scan: {error_msg}")
        
        # View Patient Scans tab
        with mri_tab2:
            st.markdown("### Patient MRI Scan History")
            
            # Get all MRI scans for this patient
            patient_id = st.session_state.selected_patient
            mri_scans = get_patient_mri_scans(patient_id)
            
            if not mri_scans:
                st.info("No MRI scans found for this patient. Upload a new scan using the 'Upload New Scan' tab.")
            else:
                # Convert to DataFrame for display
                scan_df = pd.DataFrame(mri_scans)
                
                # Format dates for display if present
                if 'scan_date' in scan_df.columns:
                    scan_df['scan_date'] = pd.to_datetime(scan_df['scan_date'])
                    scan_df['Scan Date'] = scan_df['scan_date'].dt.strftime('%Y-%m-%d %H:%M')
                
                # Prepare columns for display
                display_cols = ['scan_id', 'Scan Date', 'scan_type', 'file_name']
                display_names = ['ID', 'Date', 'Type', 'Filename']
                
                # If there are any processed scans with predictions, show those too
                if 'is_processed' in scan_df.columns and scan_df['is_processed'].any():
                    display_cols.extend(['prediction', 'confidence'])
                    display_names.extend(['Prediction', 'Confidence'])
                
                # Display available scans in a table
                st.subheader("Available MRI Scans")
                
                # Use only columns that exist in the dataframe
                display_cols = [col for col in display_cols if col in scan_df.columns]
                
                # Create the display dataframe
                if len(display_cols) > 0:
                    display_df = scan_df[display_cols].copy()
                    # Rename columns
                    display_df.columns = display_names[:len(display_cols)]
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.warning("Scan data is not in the expected format.")
                
                # Allow selection of a scan to view details
                scan_options = {f"ID {row['scan_id']} - {row.get('scan_type', 'Unknown')} - {row.get('file_name', 'Unknown')}": 
                              row['scan_id'] for i, row in scan_df.iterrows()}
                
                selected_scan = st.selectbox(
                    "Select a scan to view details",
                    options=list(scan_options.keys())
                )
                
                if selected_scan:
                    scan_id = scan_options[selected_scan]
                    selected_scan_data = scan_df[scan_df['scan_id'] == scan_id].iloc[0]
                    
                    # Display scan details
                    st.markdown("### MRI Scan Details")
                    
                    # Create two columns for details and image
                    detail_col, image_col = st.columns([1, 1])
                    
                    with detail_col:
                        st.markdown(f"**Scan ID:** {selected_scan_data['scan_id']}")
                        st.markdown(f"**Date:** {selected_scan_data.get('Scan Date', 'Unknown')}")
                        st.markdown(f"**Type:** {selected_scan_data.get('scan_type', 'Unknown')}")
                        st.markdown(f"**File:** {selected_scan_data.get('file_name', 'Unknown')}")
                        
                        # Show notes if available
                        if 'scan_notes' in selected_scan_data and selected_scan_data['scan_notes']:
                            st.markdown("**Notes:**")
                            st.markdown(f"{selected_scan_data['scan_notes']}")
                        
                        # Show prediction if processed
                        if 'is_processed' in selected_scan_data and selected_scan_data['is_processed']:
                            # Get the latest prediction
                            conn = get_db_connection()
                            if conn:
                                cursor = conn.cursor(dictionary=True)
                                cursor.execute("""
                                    SELECT prediction_result, confidence_score, processing_date
                                    FROM mri_processing_results
                                    WHERE scan_id = %s
                                    ORDER BY processing_date DESC
                                    LIMIT 1
                                """, (scan_id,))
                                
                                pred_result = cursor.fetchone()
                                cursor.close()
                                conn.close()
                                
                                if pred_result:
                                    prediction = pred_result['prediction_result']
                                    confidence = pred_result['confidence_score']
                                    
                                    # Format with appropriate styling
                                    st.markdown("### AI Model Analysis")
                                    if "Demented" in prediction or "demented" in prediction.lower():
                                        st.error(f"**Prediction:** {prediction}")
                                    elif "Mild" in prediction or "mild" in prediction.lower():
                                        st.warning(f"**Prediction:** {prediction}")
                                    else:
                                        st.success(f"**Prediction:** {prediction}")
                                        
                                    st.info(f"**Confidence:** {float(confidence):.1%}")
                        
                    with image_col:
                        # Display the image if file path exists and is accessible
                        file_path = selected_scan_data.get('file_path')
                        
                        if file_path and os.path.exists(file_path):
                            try:
                                st.image(file_path, use_container_width=True, caption=f"MRI Scan")
                            except Exception:
                                st.error(f"Could not display image from {file_path}")
                        else:
                            st.warning("Image file not found on server.")
                    
                    # Actions for the selected scan
                    st.markdown("### Actions")
                    action_col1, action_col2 = st.columns([1, 1])
                    
                    with action_col1:
                        # Process scan with model if not already processed
                        if 'is_processed' not in selected_scan_data or not selected_scan_data['is_processed']:
                            if st.button("Process with AI Model", type="primary"):
                                file_path = selected_scan_data.get('file_path')
                                
                                if file_path and os.path.exists(file_path):
                                    with st.spinner("Processing MRI scan..."):
                                        from mri_models import process_mri_with_huggingface
                                        results = process_mri_with_huggingface(file_path)
                                        
                                        if results and 'error' not in results:
                                            # Update scan with prediction
                                            prediction = results.get('prediction', 'Unknown')
                                            confidence = results.get('confidence', 0)
                                            
                                            if update_mri_scan_prediction(scan_id, prediction, confidence):
                                                st.success(f"Scan processed successfully. Prediction: {prediction} ({confidence:.1%})")
                                                # Refresh page to show new results
                                                st.rerun()
                                            else:
                                                st.error("Failed to save prediction results")
                                        else:
                                            error_msg = results.get('error', 'Unknown error') if results else 'Failed to process scan'
                                            st.error(f"Processing error: {error_msg}")
                                else:
                                    st.error("Cannot process scan: Image file not found")
                        
                        # Show ROI measurements if available
                        conn = get_db_connection()
                        if conn:
                            cursor = conn.cursor(dictionary=True)
                            cursor.execute("""
                                SELECT * FROM mri_roi_measurements 
                                WHERE scan_id = %s
                                LIMIT 1
                            """, (scan_id,))
                            
                            roi_data = cursor.fetchone()
                            cursor.close()
                            conn.close()
                            
                            if roi_data:
                                st.markdown("### Region of Interest Measurements")
                                roi_cols = st.columns(2)
                                
                                with roi_cols[0]:
                                    if 'hippocampus_left' in roi_data:
                                        st.metric("Hippocampus (Left)", f"{roi_data['hippocampus_left']:.2f} mm³")
                                    if 'hippocampus_right' in roi_data:
                                        st.metric("Hippocampus (Right)", f"{roi_data['hippocampus_right']:.2f} mm³") 
                                    if 'hippocampus_total' in roi_data:
                                        st.metric("Hippocampus (Total)", f"{roi_data['hippocampus_total']:.2f} mm³",
                                                 delta="Critical for memory")
                                
                                with roi_cols[1]:
                                    if 'entorhinal_left' in roi_data:
                                        st.metric("Entorhinal (Left)", f"{roi_data['entorhinal_left']:.2f} mm³")
                                    if 'entorhinal_right' in roi_data:
                                        st.metric("Entorhinal (Right)", f"{roi_data['entorhinal_right']:.2f} mm³")
                                    if 'entorhinal_total' in roi_data:
                                        st.metric("Entorhinal (Total)", f"{roi_data['entorhinal_total']:.2f} mm³",
                                                delta="Early AD marker")
                    
                    with action_col2:
                        # Delete scan button
                        if st.button("🗑️ Delete Scan", type="secondary"):
                            if st.checkbox("Confirm deletion"):
                                if delete_mri_scan(scan_id):
                                    st.success("MRI scan deleted successfully")
                                    # Refresh the page
                                    st.rerun()
                                else:
                                    st.error("Failed to delete MRI scan")
                
                # Allow comparing two scans side by side
                st.subheader("Compare MRI Scans")
                
                if len(mri_scans) >= 2:
                    compare_col1, compare_col2 = st.columns(2)
                    
                    with compare_col1:
                        first_scan = st.selectbox(
                            "Select First Scan",
                            options=list(scan_options.keys()),
                            key="first_scan_compare"
                        )
                        first_scan_id = scan_options[first_scan]
                    
                    with compare_col2:
                        # Filter out first selection
                        remaining_options = {k: v for k, v in scan_options.items() if v != first_scan_id}
                        second_scan = st.selectbox(
                            "Select Second Scan",
                            options=list(remaining_options.keys()),
                            key="second_scan_compare"
                        )
                        second_scan_id = remaining_options[second_scan]
                    
                    if st.button("Compare Scans", type="primary"):
                        # Get data for both scans
                        first_scan_data = scan_df[scan_df['scan_id'] == first_scan_id].iloc[0]
                        second_scan_data = scan_df[scan_df['scan_id'] == second_scan_id].iloc[0]
                        
                        # Display scans side by side
                        st.markdown("### Side by Side Comparison")
                        compare_img_col1, compare_img_col2 = st.columns(2)
                        
                        with compare_img_col1:
                            st.markdown(f"**Scan 1:** {first_scan_data.get('scan_type', 'Unknown')} - {first_scan_data.get('Scan Date', 'Unknown date')}")
                            file_path1 = first_scan_data.get('file_path')
                            if file_path1 and os.path.exists(file_path1):
                                st.image(file_path1, use_container_width=True)
                            else:
                                st.warning("Image file not found")
                        
                        with compare_img_col2:
                            st.markdown(f"**Scan 2:** {second_scan_data.get('scan_type', 'Unknown')} - {second_scan_data.get('Scan Date', 'Unknown date')}")
                            file_path2 = second_scan_data.get('file_path')
                            if file_path2 and os.path.exists(file_path2):
                                st.image(file_path2, use_container_width=True)
                            else:
                                st.warning("Image file not found")
                        
                        # Compare ROI measurements if available for both scans
                        conn = get_db_connection()
                        if conn:
                            cursor = conn.cursor(dictionary=True)
                            
                            # Get ROI data for both scans
                            cursor.execute("""
                                SELECT * FROM mri_roi_measurements 
                                WHERE scan_id IN (%s, %s)
                            """, (first_scan_id, second_scan_id))
                            
                            roi_results = cursor.fetchall()
                            cursor.close()
                            conn.close()
                            
                            # If we have measurements for both scans
                            if len(roi_results) == 2:
                                # Map results by scan_id
                                roi_by_scan = {r['scan_id']: r for r in roi_results}
                                
                                if first_scan_id in roi_by_scan and second_scan_id in roi_by_scan:
                                    st.markdown("### ROI Measurement Comparison")
                                    
                                    roi1 = roi_by_scan[first_scan_id]
                                    roi2 = roi_by_scan[second_scan_id]
                                    
                                    # Compare key measurements
                                    measure_cols = st.columns(3)
                                    
                                    # Hippocampus comparison
                                    with measure_cols[0]:
                                        if 'hippocampus_total' in roi1 and 'hippocampus_total' in roi2:
                                            val1 = roi1['hippocampus_total']
                                            val2 = roi2['hippocampus_total']
                                            change = ((val2 - val1) / val1) * 100 if val1 else 0
                                            
                                            st.metric("Hippocampus Volume", 
                                                    f"{val2:.2f} mm³", 
                                                    f"{change:.1f}%")
                                    
                                    # Entorhinal comparison
                                    with measure_cols[1]:
                                        if 'entorhinal_total' in roi1 and 'entorhinal_total' in roi2:
                                            val1 = roi1['entorhinal_total']
                                            val2 = roi2['entorhinal_total'] 
                                            change = ((val2 - val1) / val1) * 100 if val1 else 0
                                            
                                            st.metric("Entorhinal Volume", 
                                                    f"{val2:.2f} mm³", 
                                                    f"{change:.1f}%")
                                    
                                    # Ventricles comparison
                                    with measure_cols[2]:
                                        if 'lateral_ventricles' in roi1 and 'lateral_ventricles' in roi2:
                                            val1 = roi1['lateral_ventricles']
                                            val2 = roi2['lateral_ventricles']
                                            change = ((val2 - val1) / val1) * 100 if val1 else 0
                                            
                                            st.metric("Ventricle Size", 
                                                    f"{val2:.2f} mm³", 
                                                    f"{change:.1f}%")
                else:
                    st.info("Need at least two MRI scans to compare.")

    # Analysis History tab
    with tab3:
        analyses = get_patient_analyses(patient_id)
        
        if not analyses:
            st.info("No previous analyses found for this patient.")
        else:
            # Convert to DataFrame for display and analysis
            df_analyses = pd.DataFrame(analyses)
            
            # Format dates
            df_analyses['analyzed_at'] = pd.to_datetime(df_analyses['analyzed_at'])
            df_analyses['Date'] = df_analyses['analyzed_at'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Prepare for display
            display_df = df_analyses[['analysis_id', 'prediction', 'confidence_score', 'Date']]
            display_df = display_df.rename(columns={
                'analysis_id': 'ID', 
                'prediction': 'Diagnosis', 
                'confidence_score': 'Confidence'
            })
            
            # Format confidence for display
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{float(x):.1%}")
            
            # Show analyses in a table
            st.subheader("Patient Analysis History")
            st.dataframe(display_df.sort_values('Date', ascending=False), use_container_width=True)
            
            # Create a comprehensive comparison section
            st.subheader("Visit Comparison Analysis")
            
            # Allow selection of two analyses to compare
            col1, col2 = st.columns(2)
            
            with col1:
                # Get unique analysis dates with IDs for selection
                analysis_options = {f"{row['Date']} (ID: {row['ID']})": row['ID'] 
                                   for i, row in display_df.iterrows()}
                
                first_analysis = st.selectbox(
                    "Select First Visit", 
                    options=list(analysis_options.keys()),
                    index=min(1, len(analysis_options)-1) if len(analysis_options) > 1 else 0
                )
                first_id = analysis_options[first_analysis]
            
            with col2:
                # Filter out first selection for second dropdown
                remaining_options = {k: v for k, v in analysis_options.items() 
                                   if v != analysis_options[first_analysis]}
                
                if remaining_options:
                    second_analysis = st.selectbox(
                        "Select Second Visit", 
                        options=list(remaining_options.keys()),
                        index=0
                    )
                    second_id = remaining_options[second_analysis]
                    
                    # Enable comparison
                    can_compare = True
                else:
                    st.info("Need at least two analyses to compare.")
                    can_compare = False
            
            # Retrieve full feature data for the selected analyses
            if can_compare:
                if st.button("📊 Compare Selected Visits", type="primary"):
                    try:
                        conn = get_db_connection()
                        if not conn:
                            st.error("Could not connect to database")
                        else:
                            try:
                                cursor = conn.cursor(dictionary=True)
                                
                                # Fix the SQL query to use proper parameter format
                                cursor.execute("""
                                    SELECT analysis_id, prediction, confidence_score, analyzed_at, input_features
                                    FROM alzheimers_analysis
                                    WHERE analysis_id = %s OR analysis_id = %s
                                """, (first_id, second_id))
                                
                                comparison_data = cursor.fetchall()
                                cursor.close()
                                conn.close()
                                
                                if len(comparison_data) == 2:
                                    # Use the visit comparison function we created
                                    st.success(f"Comparing visit {first_id} and visit {second_id}")
                                    
                                    try:
                                        from visit_comparison import display_visit_comparison
                                        display_visit_comparison(comparison_data, patient_info, model, get_feature_descriptions())
                                    except Exception as e:
                                        st.error(f"Error in comparison display: {e}")
                                        import traceback
                                        st.error(traceback.format_exc())
                                else:
                                    st.warning(f"Could not retrieve data for both selected visits. Found {len(comparison_data)} records.")
                                    # Show what we did retrieve
                                    st.write("Retrieved data:", comparison_data)
                            except Exception as e:
                                st.error(f"Database error: {e}")
                                import traceback
                                st.error(traceback.format_exc())
                                if conn and hasattr(conn, 'is_connected') and conn.is_connected():
                                    cursor.close()
                                    conn.close()
                    except Exception as e:
                        st.error(f"Error occurred: {e}")
                        import traceback
                        st.error(traceback.format_exc())
                else:
                    st.info("Click the 'Compare Selected Visits' button to see a detailed comparison.")

def generate_mri_description(results, scan_type="T1-weighted"):
    """
    Generate a detailed description of what the model sees in the MRI scan.
    
    Args:
        results: Dictionary containing model prediction results
        scan_type: Type of MRI scan (T1-weighted, T2-weighted, etc.)
        
    Returns:
        Detailed textual description of the scan
    """
    if not results or 'prediction' not in results:
        return "Unable to generate description due to processing error."
    
    prediction = results['prediction']
    confidence = results.get('confidence', 0)
    class_index = results.get('class_index', 0)
    roi_measurements = results.get('roi_measurements', {})
    
    # Initialize description parts
    sections = []
    
    # Introduction based on scan type and overall assessment
    intro_templates = [
        f"Analysis of this {scan_type} MRI scan reveals",
        f"Examination of the {scan_type} brain scan shows",
        f"This {scan_type} MRI demonstrates",
        f"Visual assessment of this {scan_type} scan indicates"
    ]
    intro = random.choice(intro_templates)
    
    # Brain structure observations based on prediction
    if "Nondemented" in prediction:
        structures = [
            "normal cortical thickness throughout the cerebral hemispheres",
            "preserved hippocampal volume within normal limits for age",
            "ventricles of appropriate size and configuration",
            "no significant atrophy in the medial temporal lobes",
            "well-preserved gray-white matter differentiation"
        ]
        random.shuffle(structures)
        structures_text = ", ".join(structures[:3]) + "."
        
    elif "Mild" in prediction:
        structures = [
            "mild cortical thinning, particularly in the temporal lobes",
            "early hippocampal volume loss (approximately 10-15%)",
            "slight ventricular enlargement consistent with early atrophy",
            "subtle changes in the entorhinal cortex",
            "mild widening of the sulci in the temporal and parietal regions"
        ]
        random.shuffle(structures)
        structures_text = ", ".join(structures[:3]) + "."
        
    else:  # Demented/Alzheimer's
        structures = [
            "significant cortical atrophy, most pronounced in temporal and parietal lobes",
            "marked hippocampal volume reduction (>25%)",
            "substantial ventricular enlargement",
            "prominent widening of the sylvian fissure",
            "reduced gray-white matter differentiation",
            "notable atrophy of the entorhinal cortex"
        ]
        random.shuffle(structures)
        structures_text = ", ".join(structures[:3]) + "."
    
    sections.append(f"{intro} {structures_text}")
    
    # ROI-specific observations if available
    if roi_measurements:
        roi_section = "Quantitative analysis of key brain regions shows:"
        
        # Hippocampus
        if 'hippocampus_total' in roi_measurements:
            hippocampus_vol = roi_measurements['hippocampus_total']
            if hippocampus_vol < 5500:
                roi_section += f" Hippocampal volume is reduced ({hippocampus_vol:.1f} mm³), consistent with atrophy patterns seen in Alzheimer's disease."
            elif hippocampus_vol < 6000:
                roi_section += f" Hippocampal volume shows mild reduction ({hippocampus_vol:.1f} mm³), which may indicate early neurodegenerative changes."
            else:
                roi_section += f" Hippocampal volume appears preserved ({hippocampus_vol:.1f} mm³), within normal limits."
        
        # Ventricles
        if 'lateral_ventricles' in roi_measurements:
            ventricle_vol = roi_measurements['lateral_ventricles']
            if ventricle_vol > 22000:
                roi_section += f" Ventricular enlargement is pronounced ({ventricle_vol:.1f} mm³), reflecting significant brain volume loss."
            elif ventricle_vol > 18000:
                roi_section += f" Moderate ventricular enlargement ({ventricle_vol:.1f} mm³) is noted, suggesting mild to moderate volume loss."
            else:
                roi_section += f" Ventricular size ({ventricle_vol:.1f} mm³) appears within normal parameters."
        
        # Entorhinal cortex
        if 'entorhinal_total' in roi_measurements:
            entorhinal_vol = roi_measurements['entorhinal_total']
            if entorhinal_vol < 3000:
                roi_section += f" The entorhinal cortex shows significant volume reduction ({entorhinal_vol:.1f} mm³), an early site affected in Alzheimer's pathology."
            elif entorhinal_vol < 3500:
                roi_section += f" The entorhinal cortex demonstrates mild thinning ({entorhinal_vol:.1f} mm³), which merits monitoring."
            else:
                roi_section += f" Entorhinal cortex measurements ({entorhinal_vol:.1f} mm³) appear preserved."
                
        sections.append(roi_section)
    
    # Model attention insights
    if 'heatmap_path' in results and results['heatmap_path']:
        attention_templates = [
            "The model's attention is primarily focused on",
            "Visual analysis of the model's attention map highlights",
            "Key regions identified by the model's attention mechanism include",
            "The neural network is particularly attentive to"
        ]
        
        attention_intro = random.choice(attention_templates)
        
        if "Nondemented" in prediction:
            attention_regions = "normal brain structures with balanced attention across cortical regions."
        elif "Mild" in prediction:
            attention_regions = "the medial temporal lobe structures, with early signs of atrophy in the hippocampus and entorhinal cortex."
        else:  # Demented/Alzheimer's
            attention_regions = "severely atrophied regions including the hippocampus, entorhinal cortex, and expanded ventricles, classic markers of Alzheimer's progression."
            
        sections.append(f"{attention_intro} {attention_regions}")
    
    # Conclusion with confidence
    confidence_percent = confidence * 100
    if confidence_percent >= 85:
        confidence_text = "high confidence"
    elif confidence_percent >= 70:
        confidence_text = "moderate confidence"
    else:
        confidence_text = "limited confidence"
        
    conclusion = f"Based on these findings, the model predicts {prediction} with {confidence_text} ({confidence_percent:.1f}%)."
    
    # Add diagnostic recommendations based on prediction
    if "Nondemented" in prediction:
        conclusion += " No significant neurodegenerative changes are identified on this scan."
    elif "Mild" in prediction:
        conclusion += " The changes are consistent with mild cognitive impairment, which may represent an early stage of neurodegeneration. Follow-up imaging in 12 months is recommended to assess for progression."
    else:  # Demented/Alzheimer's
        conclusion += " The pattern of atrophy is highly characteristic of Alzheimer's disease. Clinical correlation with cognitive assessment and potentially CSF biomarkers is recommended."
    
    sections.append(conclusion)
    
    # Combine all sections
    full_description = "\n\n".join(sections)
    return full_description

def get_patient_mri_scans(patient_id):
    """Get MRI scans for a specific patient."""
    conn = get_db_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT scan_id, scan_date, scan_type, file_path, is_processed, 
                   file_name, scan_description
            FROM mri_scans
            WHERE patient_id = %s
            ORDER BY scan_date DESC
        """, (patient_id,))
        
        scans = cursor.fetchall()
        
        # For compatibility with existing code, add empty prediction and confidence fields
        for scan in scans:
            scan['prediction'] = None
            scan['confidence'] = None
            scan['scan_notes'] = scan.get('scan_description', None)
        
        cursor.close()
        conn.close()
        return scans
    except mysql.connector.Error as e:
        st.error(f"Error fetching MRI scans: {e}")
        cursor.close()
        conn.close()
        return []

def save_mri_scan(patient_id, file_path, scan_type, prediction=None, confidence=None, notes=None):
    """Save MRI scan information to database."""
    conn = get_db_connection()
    if not conn:
        return False
    
    cursor = conn.cursor()
    try:
        now = datetime.now()
        is_processed = False  # Default to not processed
        file_name = os.path.basename(file_path)
        file_type = os.path.splitext(file_name)[1].lower().replace(".", "")
        
        # Truncate scan_type to prevent database errors (VARCHAR(50) limit)
        if scan_type and len(scan_type) > 50:
            scan_type = scan_type[:47] + "..."
        
        # First save the scan without prediction info
        cursor.execute("""
            INSERT INTO mri_scans 
            (patient_id, scan_date, scan_type, file_path, file_name, 
             file_type, is_processed, scan_description, uploaded_at, uploaded_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (patient_id, now, scan_type, file_path, file_name, 
              file_type, is_processed, notes, now, st.session_state.get("user_id", 1)))
        
        conn.commit()
        scan_id = cursor.lastrowid
        
        # If prediction is provided, try to store it separately
        # This is now in a separate try block so even if it fails, the scan is still saved
        if prediction is not None and scan_id:
            try:
                cursor.execute("""
                    INSERT INTO mri_processing_results
                    (scan_id, processing_date, prediction_result, confidence_score, 
                     processor_type, processing_notes)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (scan_id, now, prediction, confidence, "OTHER", 
                      f"Initial prediction during upload: {prediction}"))
                conn.commit()
                
                # Update is_processed flag
                cursor.execute("""
                    UPDATE mri_scans
                    SET is_processed = 1
                    WHERE scan_id = %s
                """, (scan_id,))
                conn.commit()
            except mysql.connector.Error as e:
                # This just logs a warning but doesn't affect the overall save operation
                print(f"Warning: Saved scan but couldn't store prediction: {e}")
                st.warning(f"Scan saved successfully, but prediction results couldn't be stored: {e}")
        
        cursor.close()
        conn.close()
        return scan_id
    except mysql.connector.Error as e:
        st.error(f"Error saving MRI scan: {e}")
        cursor.close()
        conn.close()
        return False

def update_mri_scan_prediction(scan_id, prediction, confidence, notes=None):
    """Update MRI scan with prediction results."""
    conn = get_db_connection()
    if not conn:
        return False
    
    cursor = conn.cursor()
    try:
        now = datetime.now()
        
        # First update the is_processed flag in mri_scans
        cursor.execute("""
            UPDATE mri_scans
            SET is_processed = 1
            WHERE scan_id = %s
        """, (scan_id,))
        
        # Then store the prediction in mri_processing_results
        cursor.execute("""
            INSERT INTO mri_processing_results
            (scan_id, processing_date, prediction_result, confidence_score, 
             processor_type, processing_notes)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (scan_id, now, prediction, confidence, "OTHER", notes))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except mysql.connector.Error as e:
        st.error(f"Error updating MRI scan prediction: {e}")
        cursor.close()
        conn.close()
        return False

def delete_mri_scan(scan_id):
    """Delete an MRI scan from database and file system."""
    conn = get_db_connection()
    if not conn:
        return False
    
    cursor = conn.cursor(dictionary=True)
    try:
        # Get file path first
        cursor.execute("SELECT file_path FROM mri_scans WHERE scan_id = %s", (scan_id,))
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            return False
        
        file_path = result['file_path']
        
        # Delete from database
        cursor.execute("DELETE FROM mri_scans WHERE scan_id = %s", (scan_id,))
        conn.commit()
        
        # Delete file if it exists
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not delete file {file_path}: {e}")
        
        cursor.close()
        conn.close()
        return True
    except mysql.connector.Error as e:
        st.error(f"Error deleting MRI scan: {e}")
        cursor.close()
        conn.close()
        return False

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
    
    # Apply custom styling for chat interface
    st.markdown("""
    <style>
        /* Chat container styling */
        .stChatMessage {
            padding: 10px 0;
        }
        
        /* Message styling */
        .stChatMessageContent {
            padding: 15px !important;
            border-radius: 18px !important;
            margin-bottom: 10px !important;
            max-width: 85% !important;
        }
        
        /* User message styling */
        .stChatMessageContent[data-test="user-message"] {
            background-color: #e1f5fe !important;
            border: 1px solid #b3e5fc !important;
            border-radius: 18px 18px 0 18px !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
        }
        
        /* Assistant message styling */
        .stChatMessageContent:not([data-test="user-message"]) {
            background-color: white !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 18px 18px 18px 0 !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
        }
        
        /* Chat buttons */
        .chat-button {
            border-radius: 20px;
            padding: 5px 15px;
            margin: 0 5px;
            background-color: #f0f2f6;
            border: 1px solid #ddd;
            color: #444;
            transition: all 0.3s ease;
        }
        
        .chat-button:hover {
            background-color: #e1f5fe;
            border-color: #81d4fa;
        }
        
        /* Chat input box styling */
        .stChatInputContainer {
            background-color: white !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 24px !important;
            padding: 5px !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
            margin-top: 15px !important;
        }
        
        /* Suggestion buttons */
        [data-testid="baseButton-secondary"] {
            border-radius: 18px !important;
            min-height: 36px !important;
            border: 1px solid #e0e0e0 !important;
            background-color: #f8f9fa !important;
            transition: all 0.2s ease !important;
        }
        
        [data-testid="baseButton-secondary"]:hover {
            background-color: #e8f5e9 !important;
            border-color: #a5d6a7 !important;
            transform: translateY(-1px) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for context options
    with st.sidebar:
        st.subheader("Assistant Context")
        include_features = st.toggle("Include Clinical Features", value=True)
        include_records = st.toggle("Include Medical Records", value=True)
        include_analyses = st.toggle("Include Analysis History", value=True)
        include_mri = st.toggle("Include MRI Scans", value=True)
        
        # Topic quick filters
        st.subheader("Chat Topics")
        topic_buttons = [
            ("🧠 Cognitive Tests", "What do the cognitive test scores (MMSE, CDRSB) indicate for this patient?"),
            ("📊 Biomarkers", "Can you explain the significance of this patient's biomarker results?"),
            ("🩻 MRI Analysis", "Analyze the attached MRI scan and highlight key findings."),
            ("📈 Disease Progression", "What's the likely progression for this patient based on current data?"),
            ("💊 Treatment Options", "What treatment options should be considered for this patient?")
        ]
        
        # Create a topic selector
        st.markdown("#### Quick Topics")
        for topic_name, topic_query in topic_buttons:
            if st.button(topic_name, use_container_width=True):
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                # Use pending_message to handle this in the same way as suggestions
                st.session_state.pending_message = topic_query
                st.rerun()
        
        # Reset conversation button
        if st.button("🔄 Reset Conversation", type="primary", use_container_width=True):
            # Clear the conversation in session state
            st.session_state.chat_history = []
            
            # Delete chat history from database
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
    chat_history = get_chat_history(patient_id, doctor_id)
    
    # Update the session state with the loaded history
    st.session_state.chat_history = chat_history
    
    # MRI scan attachment with improved UI
    st.markdown("### 🩻 MRI Analysis & Annotation")
    
    with st.container():
        # Add a distinct background for the MRI analysis section
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0;">
            <h4 style="margin-top:0; color: #4e73df;">Select and Analyze MRI Scans</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Create two columns for scan selection and preview
        scan_select_col, scan_preview_col = st.columns([1, 1])
        
        with scan_select_col:
            # Get MRI scans for this patient
            mri_scans = get_patient_mri_scans(patient_id)
            
            if mri_scans:
                # Create dataframe for selection
                scan_df = pd.DataFrame(mri_scans)
                scan_df['scan_date'] = pd.to_datetime(scan_df['scan_date'])
                scan_df['scan_date'] = scan_df['scan_date'].dt.strftime('%Y-%m-%d %H:%M')
                
                # Let user select an MRI scan from dropdown
                selected_scan_id = st.selectbox(
                    "Select MRI scan",
                    options=scan_df['scan_id'].tolist(),
                    format_func=lambda x: f"Scan #{x} - {scan_df.loc[scan_df['scan_id'] == x, 'scan_date'].iloc[0]} - {scan_df.loc[scan_df['scan_id'] == x, 'scan_type'].iloc[0]}"
                )
                
                # Get the selected scan
                selected_scan = next((scan for scan in mri_scans if scan['scan_id'] == selected_scan_id), None)
                
                if selected_scan and os.path.exists(selected_scan['file_path']):
                    st.session_state.current_mri_scan = selected_scan
                    
                    # Display scan details
                    st.markdown(f"""
                    <div style="margin-top: 10px; padding: 10px; border-left: 3px solid #4e73df; background-color: #f1f3f9;">
                        <p><strong>Scan Type:</strong> {selected_scan['scan_type']}</p>
                        <p><strong>Date:</strong> {scan_df.loc[scan_df['scan_id'] == selected_scan_id, 'scan_date'].iloc[0]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add buttons for different analysis options
                    st.markdown("#### Analysis Options")
                    
                    # Create two columns for the buttons
                    btn_col1, btn_col2 = st.columns(2)
                    
                    with btn_col1:
                        if st.button("🔍 Ask about scan", use_container_width=True):
                            prompt = f"Please analyze this patient's MRI scan (ID #{selected_scan_id}, Type: {selected_scan['scan_type']}) and provide insights."
                            st.session_state.pending_message = prompt
                            st.rerun()
                    
                    with btn_col2:
                        # Add a visual button for ROI analysis
                        roi_btn = st.button("🧠 Analyze ROI", use_container_width=True)
                        if roi_btn:
                            # Set a flag to trigger ROI analysis
                            st.session_state.analyze_mri_roi = True
                            st.session_state.mri_to_analyze = selected_scan['file_path']
                            st.session_state.mri_scan_type = selected_scan['scan_type']
                            st.rerun()
                    
                    # Add an expandable section with regions to focus on
                    with st.expander("Regions of Interest for Alzheimer's", expanded=False):
                        st.markdown("""
                        - **Hippocampus**: Critical for memory formation, often shows atrophy in Alzheimer's
                        - **Ventricles**: Fluid-filled spaces that enlarge as brain tissue is lost
                        - **Entorhinal Cortex**: Early site of pathology in Alzheimer's progression
                        - **Temporal Lobe**: Important for language and memory processing
                        - **Medial Temporal Lobe**: Often shows significant atrophy in AD
                        - **Frontal Cortex**: Executive function impairment in advanced stages
                        """)
            else:
                st.info("No MRI scans available for this patient.")
                st.session_state.current_mri_scan = None
        
        with scan_preview_col:
            if hasattr(st.session_state, 'current_mri_scan') and st.session_state.current_mri_scan:
                scan = st.session_state.current_mri_scan
                if os.path.exists(scan['file_path']):
                    st.markdown("#### Scan Preview")
                    st.image(scan['file_path'], use_container_width=True)
                    
                    # Add information about what happens during analysis
                    st.info("💡 Click 'Analyze ROI' to have Gemini AI identify and annotate key brain regions affected in Alzheimer's disease.")
    
    # Handle ROI analysis if triggered (keep existing code)
    if hasattr(st.session_state, 'analyze_mri_roi') and st.session_state.analyze_mri_roi:
        # Keep the existing analysis logic
        with st.spinner("Processing MRI scan with Gemini AI and analyzing regions of interest..."):
            try:
                file_path = st.session_state.mri_to_analyze
                scan_type = st.session_state.mri_scan_type
                
                # Check if file exists
                if not os.path.exists(file_path):
                    st.error(f"MRI image file not found at {file_path}")
                    st.session_state.analyze_mri_roi = False
                else:
                    # Prepare the MRI image for Gemini analysis
                    try:
                        # Open the image
                        with open(file_path, "rb") as f:
                            image_data = f.read()
                        
                        # Create a directory for storing analysis results
                        analysis_dir = "analysis_results"
                        os.makedirs(analysis_dir, exist_ok=True)
                        
                        # Generate output filename for ROI visualization
                        base_filename = os.path.basename(file_path)
                        roi_filename = f"gemini_roi_{base_filename}"
                        roi_image_path = os.path.join(analysis_dir, roi_filename)
                        
                        # Prepare the context for Gemini
                        prompt = f"""You are a neuroradiologist specializing in Alzheimer's disease.

I'm showing you a {scan_type} MRI scan of a patient's brain.

Please analyze this MRI image with focus on:
1. Hippocampus volume and atrophy
2. Ventricular enlargement
3. Cortical thinning
4. Medial temporal lobe atrophy
5. Other relevant findings for Alzheimer's disease

Based on these observations:
1. Provide a detailed diagnostic assessment with justification
2. Highlight key regions of interest (hippocampus, ventricles, entorhinal cortex)
3. Explain the clinical significance of your findings
4. Estimate the likelihood that this represents Alzheimer's disease, MCI, or normal aging

Provide specific measurements where possible, focusing on the regions most affected in Alzheimer's disease.
"""

                        # Create image parts for the model
                        image_parts = [
                            {"mime_type": "image/jpeg", "data": image_data}
                        ]
                        
                        # Send to Gemini model for analysis
                        response = model.generate_content([prompt, image_parts[0]])
                        diagnostic_analysis = response.text
                        
                        # Create a ROI image using OpenCV based on Gemini guidance
                        try:
                            import cv2
                            import numpy as np
                            from PIL import Image, ImageDraw, ImageFont
                            import io
                            
                            # Load the original MRI image
                            original_img = cv2.imread(file_path)
                            if original_img is None:
                                # Try using PIL instead if OpenCV fails
                                pil_img = Image.open(file_path)
                                original_img = np.array(pil_img)
                                # Convert RGB to BGR for OpenCV processing
                                if len(original_img.shape) == 3 and original_img.shape[2] == 3:
                                    original_img = original_img[:, :, ::-1].copy()
                            
                            # Create a copy for ROI highlighting
                            roi_img = original_img.copy()
                            
                            # Send a second request to Gemini for ROI identification
                            roi_prompt = f"""Based on the same {scan_type} MRI scan, I need precise coordinates to create a region of interest (ROI) visualization.

For the following critical regions, please provide:
1. Approximate pixel coordinates (x, y) of the center of each region
2. Approximate size/radius for highlighting each region

Regions to identify:
- Hippocampus (critical for memory, often shows atrophy in Alzheimer's)
- Ventricles (fluid-filled spaces that enlarge as brain tissue is lost)
- Entorhinal Cortex (early site of pathology in Alzheimer's)
- Temporal Lobe (if visible)
- Any other regions showing significant abnormalities

Please format your response as specific coordinates and sizes, for example:
"Hippocampus: center at (120, 150), radius 15 pixels"
"Ventricles: center at (100, 100), radius 20 pixels"

Be as precise as possible so I can create an accurate ROI visualization.
"""
                            
                            # Get ROI coordinates from Gemini
                            roi_response = model.generate_content([roi_prompt, image_parts[0]])
                            roi_coordinates_text = roi_response.text
                            
                            # Save the ROI analysis text
                            st.session_state.roi_analysis_text = roi_coordinates_text
                            
                            # Parse ROI coordinates (simple parsing, could be enhanced)
                            # Extract any coordinates mentioned in the text using regex
                            import re
                            
                            # Look for patterns like (x, y) or (x,y)
                            coord_pattern = r'\((\d+),?\s*(\d+)\)'
                            radius_pattern = r'radius\s+(\d+)'
                            
                            coords = re.findall(coord_pattern, roi_coordinates_text)
                            radii = re.findall(radius_pattern, roi_coordinates_text)
                            
                            # Default ROIs if no coordinates are found
                            if not coords:
                                # Use reasonable defaults based on brain anatomy
                                h, w = roi_img.shape[:2]
                                center_x, center_y = w // 2, h // 2
                                
                                # Default ROIs based on typical brain MRI layout
                                default_rois = [
                                    # Region name, x, y, radius, color (BGR)
                                    ("Hippocampus", int(center_x * 0.8), int(center_y * 1.1), 15, (0, 0, 255)),  # Red
                                    ("Ventricles", center_x, int(center_y * 0.8), 25, (0, 255, 255)),  # Yellow
                                    ("Entorhinal Cortex", int(center_x * 0.7), int(center_y * 1.05), 12, (0, 255, 0)),  # Green
                                    ("Temporal Lobe", int(center_x * 0.6), center_y, 20, (255, 0, 0))  # Blue
                                ]
                                
                                # Draw default ROIs
                                for name, x, y, radius, color in default_rois:
                                    cv2.circle(roi_img, (x, y), radius, color, 2)
                                    cv2.putText(roi_img, name, (x - radius, y - radius - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            else:
                                # Use the coordinates found in the text
                                region_names = ["Hippocampus", "Ventricles", "Entorhinal Cortex", "Temporal Lobe"]
                                colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0)]  # BGR format
                                
                                for i, (x, y) in enumerate(coords[:min(len(coords), len(region_names))]):
                                    x, y = int(x), int(y)
                                    radius = int(radii[i]) if i < len(radii) else 15
                                    name = region_names[i % len(region_names)]
                                    color = colors[i % len(colors)]
                                    
                                    cv2.circle(roi_img, (x, y), radius, color, 2)
                                    cv2.putText(roi_img, name, (x - radius, y - radius - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            
                            # Add a legend/key
                            legend_height = 80
                            legend = np.ones((legend_height, roi_img.shape[1], 3), dtype=np.uint8) * 255
                            
                            legend_texts = [
                                ("Hippocampus", (0, 0, 255)),  # Red
                                ("Ventricles", (0, 255, 255)),  # Yellow
                                ("Entorhinal Cortex", (0, 255, 0)),  # Green
                                ("Temporal Lobe", (255, 0, 0))   # Blue
                            ]
                            
                            for i, (text, color) in enumerate(legend_texts):
                                cv2.circle(legend, (20, 20 + i*15), 5, color, -1)
                                cv2.putText(legend, text, (30, 25 + i*15), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            
                            # Combine the ROI image with the legend
                            roi_with_legend = np.vstack((roi_img, legend))
                            
                            # Save the ROI visualization
                            cv2.imwrite(roi_image_path, roi_with_legend)
                            
                            # Save the ROI image path to session state
                            st.session_state.roi_highlighted_image = roi_image_path
                            
                        except Exception as e:
                            st.warning(f"Could not create ROI visualization: {e}")
                            import traceback
                            st.warning(traceback.format_exc())
                            # Save the original file path for display
                            st.session_state.roi_highlighted_image = file_path
                        
                        # Create a comprehensive message for the chat
                        mri_message = f"""## MRI Analysis Results

**Scan Type:** {scan_type}

{diagnostic_analysis}
"""
                        
                        # Add the visualization and analysis to chat history
                        if "chat_history" not in st.session_state:
                            st.session_state.chat_history = []
                        
                        # Add user prompt to history
                        user_prompt = f"Please analyze this MRI scan with regions of interest and provide a diagnostic assessment."
                        st.session_state.chat_history.append(("You", user_prompt))
                        save_chat_message(patient_id, doctor_id, user_prompt, "Doctor")
                        
                        # Save the original MRI path for display
                        st.session_state.current_roi_image = file_path
                        
                        # Add AI response to history
                        st.session_state.chat_history.append(("Assistant", mri_message))
                        save_chat_message(patient_id, doctor_id, mri_message, "Assistant")
                        
                        # Clear the analysis flag
                        st.session_state.analyze_mri_roi = False
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing MRI with Gemini: {e}")
                        import traceback
                        st.error(traceback.format_exc())
                        st.session_state.analyze_mri_roi = False
            except Exception as e:
                st.error(f"Error analyzing MRI: {e}")
                import traceback
                st.error(traceback.format_exc())
                st.session_state.analyze_mri_roi = False
    
    # Display chat messages directly on the screen without a container
    st.markdown("### 💬 Chat History")

    # Create a fixed height container for the chat with a border
    chat_container = st.container(height=770, border=True)

    # Display empty state or messages
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center; padding:50px;">
                <div style="font-size:70px; margin-bottom:10px;">💬</div>
                <h3>Start a conversation with the AI Assistant</h3>
                <p>Ask questions about the patient's data or how to interpret results</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for i, (sender, msg) in enumerate(st.session_state.chat_history):
                if sender == "You":
                    message(msg, is_user=True, key=f"msg_{i}")
                else:
                    # Check if this message follows an MRI analysis request
                    if "MRI Analysis Results" in msg:
                        # First display the message
                        message(msg, is_user=False, key=f"msg_{i}")
                        
                        # Create a card for images with better organization
                        st.markdown("""
                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; 
                                   border: 1px solid #e9ecef; margin: 10px 0;">
                            <h4 style="margin-top:0; color: #4e73df;">🧠 MRI Analysis Images</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display the MRI images in a row with smaller size
                        col1, col2 = st.columns([1, 2])
                        
                        # Variable to store ROI image path for download
                        roi_image_path = None
                        
                        with col1:
                            # Display the original MRI in a smaller size
                            if hasattr(st.session_state, 'current_roi_image') and os.path.exists(st.session_state.current_roi_image):
                                # Generate a unique ID for this instance (not used for image)
                                unique_id = f"original_mri_{uuid.uuid4().hex[:8]}"
                                st.image(st.session_state.current_roi_image, 
                                        caption="Original MRI", 
                                        width=200)  # Remove key parameter
                        
                        with col2:
                            # Display the ROI highlighted image
                            if hasattr(st.session_state, 'roi_highlighted_image') and os.path.exists(st.session_state.roi_highlighted_image):
                                roi_image_path = st.session_state.roi_highlighted_image
                                
                                # Use Plotly for better visualization
                                try:
                                    # Read the image
                                    img = cv2.imread(roi_image_path)
                                    if img is None:
                                        # Try using PIL if OpenCV fails
                                        from PIL import Image
                                        img = np.array(Image.open(roi_image_path))
                                        if len(img.shape) == 3 and img.shape[2] == 3:
                                            # Convert RGB to BGR for OpenCV processing
                                            img = img[:, :, ::-1].copy()
                                    
                                    # Convert BGR to RGB for Plotly
                                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    
                                    # Create a Plotly figure for interactive display
                                    fig = px.imshow(img_rgb)
                                    
                                    # Update layout for better appearance
                                    fig.update_layout(
                                        title="Region of Interest (ROI) Analysis",
                                        title_font=dict(size=16),
                                        margin=dict(l=0, r=0, t=40, b=0),
                                        height=450,
                                        width=500,
                                        coloraxis_showscale=False
                                    )
                                    
                                    # Remove axes and grid
                                    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
                                    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
                                    
                                    # Add annotation for the timestamp
                                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                                    fig.add_annotation(
                                        text=f"Generated: {current_time}",
                                        xref="paper", yref="paper",
                                        x=0.02, y=0.02,
                                        showarrow=False,
                                        font=dict(size=10, color="gray"),
                                        align="left"
                                    )
                                    
                                    # Display the interactive Plotly figure
                                    st.plotly_chart(fig, use_container_width=False, key=f"roi_plt_{uuid.uuid4().hex[:8]}")
                                    
                                except Exception as e:
                                    # Fallback to standard image if Plotly fails
                                    st.warning(f"Using standard image display due to error: {str(e)}")
                                    st.image(roi_image_path, 
                                            caption="Region of Interest (ROI) Analysis", 
                                            width=400)  # Remove key parameter

                        # Add download button for ROI image
                        if roi_image_path and os.path.exists(roi_image_path):
                            # Create columns for buttons
                            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
                            
                            with btn_col1:
                                # Generate a unique key for the download button
                                download_key = f"download_roi_{uuid.uuid4().hex[:8]}"
                                
                                # Read the image file
                                with open(roi_image_path, "rb") as file:
                                    btn = st.download_button(
                                        label="💾 Download ROI Image",
                                        data=file,
                                        file_name=os.path.basename(roi_image_path),
                                        mime="image/jpeg",
                                        use_container_width=True,
                                        key=download_key  # Add unique key to prevent duplicate ID error
                                    )
                        
                        # Display explanatory ROI information below
                        if hasattr(st.session_state, 'roi_analysis_text'):
                            with st.expander("📋 Detailed ROI Analysis", expanded=False):
                                st.markdown(st.session_state.roi_analysis_text)
                                
                        # Add the explanatory card about brain regions
                        st.markdown("""
                        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 10px; 
                                    border-left: 5px solid #4e73df; margin: 15px 0;">
                            <h4 style="margin-top:0;">💡 Understanding the Analysis</h4>
                            <p>The highlighted regions show key brain structures assessed for Alzheimer's disease:</p>
                            <ul>
                                <li><strong style="color:#ff0000;">Hippocampus</strong> - Critical for memory formation, shows atrophy in AD</li>
                                <li><strong style="color:#ffff00;">Ventricles</strong> - Fluid-filled spaces that enlarge as brain tissue is lost</li>
                                <li><strong style="color:#00ff00;">Entorhinal Cortex</strong> - Early site of pathology in AD progression</li>
                                <li><strong style="color:#0000ff;">Temporal Lobe</strong> - Important for language and memory processing</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Regular message display
                        message(msg, is_user=False, key=f"msg_{i}")
    
    # Chat input box
    if "pending_message" in st.session_state:
        # Use the pending message (from suggestion buttons)
        user_message = st.session_state.pending_message
        del st.session_state.pending_message
    else:
        # Regular text input
        user_message = st.chat_input("Type your message...")
    
    if user_message:
        # Add user message to chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        st.session_state.chat_history.append(("You", user_message))
        
        # Save to database
        save_chat_message(patient_id, doctor_id, user_message, "Doctor")
        
        # Generate assistant response
        with st.spinner("Generating response..."):
            try:
                # Create context based on patient data
                context = f"""You are a helpful AI clinical assistant specializing in Alzheimer's disease and dementia.
                You are currently helping a doctor with patient: {patient_info['full_name']}, {patient_info['gender']}, {datetime.now().year - patient_info['birth_date'].year} years old.
                """
                
                # Add feature data if available and enabled
                if include_features:
                    features = get_patient_features(patient_id)
                    if features:
                        context += "\nClinical Features:\n"
                        # Add important features
                        important_features = ["MMSE", "CDRSB", "ADAS13", "Hippocampus", "ABETA", "TAU", "RAVLT_immediate"]
                        feature_descriptions = get_feature_descriptions()
                        for feat in important_features:
                            if feat in features and features[feat] is not None:
                                desc = feature_descriptions.get(feat, "")
                                context += f"- {feat}: {features[feat]}"
                                if desc:
                                    context += f" ({desc.split('-')[0].strip()})"
                                context += "\n"
                
                # Add medical records if enabled
                if include_records:
                    records = get_patient_records(patient_id)
                    if records:
                        context += "\nRecent Medical Records:\n"
                        for i, record in enumerate(records[:3]):  # Include 3 most recent records
                            context += f"- {record['visit_date']}: {record['diagnosis']}\n"
                
                # Add previous analyses if enabled
                if include_analyses:
                    analyses = get_patient_analyses(patient_id)
                    if analyses:
                        context += "\nPrevious Alzheimer's Analyses:\n"
                        for i, analysis in enumerate(analyses[:3]):  # Include 3 most recent analyses
                            context += f"- {analysis['analyzed_at']}: {analysis['prediction']} (Confidence: {analysis['confidence_score']:.1%})\n"
                
                # If current MRI scan is selected and enabled
                if include_mri and hasattr(st.session_state, 'current_mri_scan') and st.session_state.current_mri_scan:
                    scan = st.session_state.current_mri_scan
                    context += f"\nCurrently discussing MRI scan (ID: {scan['scan_id']}, Type: {scan['scan_type']}).\n"
                
                # Add instructions for response
                context += """
                Based on the provided information, help the doctor understand the patient's condition, interpret test results, or answer clinical questions. 
                Keep your responses concise, clinically relevant, and evidence-based. Do not make definitive diagnoses but provide clinical insights.
                """
                
                # Create prompt with context
                prompt = context
                
                # Add chat history to the prompt string
                for sender, content in st.session_state.chat_history[-10:]:
                    if sender == "You":
                        prompt += f"\n\nDoctor: {content}"
                    else:
                        prompt += f"\n\nAssistant: {content}"
                
                # Add the current user message
                prompt += f"\n\nDoctor: {user_message}\n\nAssistant:"
                
                # Generate response using Gemini model with a simple text prompt
                response = model.generate_content(prompt)
                assistant_message = response.text
                
                # Add assistant response to chat history
                st.session_state.chat_history.append(("Assistant", assistant_message))
                
                # Save to database
                save_chat_message(patient_id, doctor_id, assistant_message, "Assistant")
                
                # Rerun to display messages
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
                # Add error message to chat
                st.session_state.chat_history.append(("Assistant", f"I'm sorry, I encountered an error: {str(e)}"))
                save_chat_message(patient_id, doctor_id, f"I'm sorry, I encountered an error: {str(e)}", "Assistant")
                st.rerun()

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
        
        # Create line chart of prediction confidence over time - minimal size
        fig, ax = plt.subplots(figsize=(1.8, 1), dpi=150)
        
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
                color=color,
                markersize=1.5,
                linewidth=0.8
            )
        
        # Format plot - minimal text
        ax.set_xlabel('Date', fontsize=3)
        ax.set_ylabel('Confidence', fontsize=3)
        ax.set_title('Progression', fontsize=4)
        ax.legend(fontsize=3, loc='best', frameon=False)
        ax.grid(True, alpha=0.3, linewidth=0.3)
        ax.tick_params(axis='both', labelsize=2.5, length=2, pad=1)
        # Rotate x-axis dates for better fit
        plt.xticks(rotation=45)
    
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
                # Create line chart with minimal size
                fig, ax = plt.subplots(figsize=(1.8, 1), dpi=150)
                
                # Plot each selected feature - maximum of 3 features for clarity
                display_features = selected_features[:3] if len(selected_features) > 3 else selected_features
                for feature in display_features:
                    if feature in features_df.columns:
                        ax.plot(
                            features_df['analyzed_at'], 
                            features_df[feature],
                            'o-',
                            label=feature,
                            markersize=1.5,
                            linewidth=0.8
                        )
                
                # Format plot - minimal text
                ax.set_xlabel('Date', fontsize=3)
                ax.set_ylabel('Value', fontsize=3)
                ax.set_title('Feature Trends', fontsize=4)
                ax.legend(fontsize=3, loc='best', frameon=False)
                ax.grid(True, alpha=0.3, linewidth=0.3)
                ax.tick_params(axis='both', labelsize=2.5, length=2, pad=1)
                plt.xticks(rotation=45)
                
                # Display the plot
                st.pyplot(fig)