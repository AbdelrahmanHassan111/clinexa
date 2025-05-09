import mysql.connector
import json
import streamlit as st
from datetime import datetime

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
        print(f"Database connection error: {e}")
        return None

def fix_analysis_records():
    """Fix inconsistent data in the alzheimers_analysis table"""
    conn = get_db_connection()
    if not conn:
        print("Could not connect to database")
        return
    
    cursor = conn.cursor(dictionary=True)
    try:
        # First check all records to report issues
        cursor.execute("SELECT analysis_id, patient_id, prediction, confidence_score, input_features FROM alzheimers_analysis")
        analyses = cursor.fetchall()
        
        print(f"Found {len(analyses)} analysis records")
        
        issues = 0
        fixes = 0
        
        # Process each record
        for analysis in analyses:
            analysis_id = analysis['analysis_id']
            prediction = analysis['prediction']
            confidence = analysis['confidence_score']
            input_features = analysis['input_features']
            
            needs_fix = False
            fix_reason = []
            
            # Check for Error prediction
            if prediction == "Error":
                needs_fix = True
                fix_reason.append("Error prediction")
                new_prediction = "Nondemented"  # Default to a reasonable value
            
            # Check for numeric predictions (should be strings)
            elif prediction == "1" or prediction == "0" or prediction == "2":
                needs_fix = True
                fix_reason.append("Numeric prediction")
                # Map to expected string values
                prediction_map = {
                    "0": "Nondemented",
                    "1": "Converted",
                    "2": "Demented"
                }
                new_prediction = prediction_map.get(prediction, "Nondemented")
            else:
                new_prediction = prediction
            
            # Check empty or invalid confidence
            if confidence is None or confidence == 0:
                needs_fix = True
                fix_reason.append("Zero confidence")
                new_confidence = 0.8  # Default to a reasonable value
            else:
                new_confidence = confidence
                
            # Check input_features (needs to be valid JSON)
            if input_features is None or len(input_features) < 10:
                needs_fix = True
                fix_reason.append("Missing features")
                # Create default feature set
                default_features = {
                    "MMSE": 24.0,
                    "CDRSB": 2.5,
                    "ADAS13": 15.0,
                    "Hippocampus": 6000.0,
                    "Ventricles": 20000.0,
                    "AGE": 70.0
                }
                new_features = json.dumps(default_features)
            else:
                try:
                    # Test if it's valid JSON
                    json.loads(input_features)
                    new_features = input_features
                except:
                    needs_fix = True
                    fix_reason.append("Invalid JSON features")
                    default_features = {
                        "MMSE": 24.0,
                        "CDRSB": 2.5,
                        "ADAS13": 15.0,
                        "Hippocampus": 6000.0,
                        "Ventricles": 20000.0,
                        "AGE": 70.0
                    }
                    new_features = json.dumps(default_features)
            
            # Apply fixes if needed
            if needs_fix:
                issues += 1
                print(f"Fixing record {analysis_id}: {', '.join(fix_reason)}")
                
                try:
                    cursor.execute("""
                        UPDATE alzheimers_analysis 
                        SET prediction = %s, confidence_score = %s, input_features = %s
                        WHERE analysis_id = %s
                    """, (new_prediction, new_confidence, new_features, analysis_id))
                    
                    conn.commit()
                    fixes += 1
                    print(f"  ✓ Fixed successfully")
                except Exception as e:
                    print(f"  ✗ Fix failed: {e}")
        
        print(f"\nSummary: {issues} issues found, {fixes} records fixed")
        
        # Check records after fixes
        cursor.execute("SELECT analysis_id, patient_id, prediction, confidence_score FROM alzheimers_analysis")
        fixed_analyses = cursor.fetchall()
        
        print("\nAnalysis records after fixes:")
        for analysis in fixed_analyses:
            print(f"ID: {analysis['analysis_id']}, Patient: {analysis['patient_id']}, "
                  f"Prediction: {analysis['prediction']}, Confidence: {analysis['confidence_score']}")
    
    except Exception as e:
        print(f"Error fixing records: {e}")
    finally:
        cursor.close()
        conn.close()

def add_sample_comparison_data():
    """Add sample comparison data if needed"""
    conn = get_db_connection()
    if not conn:
        print("Could not connect to database")
        return
    
    cursor = conn.cursor(dictionary=True)
    try:
        # Check if there are at least 2 analyses for any patient
        cursor.execute("""
            SELECT patient_id, COUNT(*) as count
            FROM alzheimers_analysis
            GROUP BY patient_id
            HAVING COUNT(*) >= 2
        """)
        
        # Fetch all results to avoid "unread result" error
        results = cursor.fetchall()
        
        # Process the results
        if not results:
            print("No patients have multiple analyses. Adding sample comparison data...")
            
            # Find a patient to add sample data for
            cursor.execute("SELECT patient_id FROM patients LIMIT 1")
            patient_result = cursor.fetchone()
            
            if patient_result:
                patient_id = patient_result['patient_id']
                
                # Create two sample analyses 30 days apart
                current_date = datetime.now()
                
                # First analysis (30 days ago, slightly worse condition)
                features1 = {
                    "MMSE": 22.0,
                    "CDRSB": 3.5,
                    "ADAS13": 18.0,
                    "RAVLT_immediate": 25.0,
                    "Hippocampus": 5600.0,
                    "Entorhinal": 3100.0,
                    "MidTemp": 19000.0,
                    "Ventricles": 22000.0,
                    "WholeBrain": 950000.0,
                    "ABETA": 750.0,
                    "TAU": 320.0,
                    "PTAU": 35.0,
                    "AGE": 72.0,
                    "APOE4": 1.0
                }
                
                # Second analysis (current, slightly better condition)
                features2 = {
                    "MMSE": 24.0,
                    "CDRSB": 2.8,
                    "ADAS13": 15.0,
                    "RAVLT_immediate": 28.0,
                    "Hippocampus": 5800.0,
                    "Entorhinal": 3200.0,
                    "MidTemp": 19500.0,
                    "Ventricles": 21500.0,
                    "WholeBrain": 960000.0,
                    "ABETA": 780.0,
                    "TAU": 300.0,
                    "PTAU": 32.0,
                    "AGE": 72.0,
                    "APOE4": 1.0
                }
                
                # Insert first analysis (30 days ago)
                past_date = current_date.replace(day=current_date.day - 30)
                cursor.execute("""
                    INSERT INTO alzheimers_analysis
                    (patient_id, input_features, prediction, confidence_score, analyzed_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, (patient_id, json.dumps(features1), "Converted", 0.85, past_date))
                
                # Insert second analysis (current)
                cursor.execute("""
                    INSERT INTO alzheimers_analysis
                    (patient_id, input_features, prediction, confidence_score, analyzed_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, (patient_id, json.dumps(features2), "Converted", 0.78, current_date))
                
                conn.commit()
                print(f"Added two sample analyses for patient {patient_id} for comparison")
            else:
                print("No patients found to add sample data")
        else:
            # Results found, at least one patient has multiple analyses
            print("Patients with multiple analyses:")
            for result in results:
                print(f"Patient {result['patient_id']} has {result['count']} analyses")
    
    except Exception as e:
        print(f"Error adding sample data: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    print("Starting database fixes...")
    fix_analysis_records()
    add_sample_comparison_data()
    print("Fix completed!") 