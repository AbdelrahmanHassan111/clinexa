import mysql.connector
import json
from datetime import datetime, timedelta
import random

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

def create_test_analyses():
    """Create test analyses for all patients to ensure history view has data"""
    conn = get_db_connection()
    if not conn:
        print("Could not connect to database")
        return
    
    cursor = conn.cursor()
    try:
        # Get all patients
        cursor.execute("SELECT patient_id, full_name FROM patients")
        patients = cursor.fetchall()
        
        if not patients:
            print("No patients found in the database")
            return
        
        print(f"Found {len(patients)} patients")
        
        # Create 3 analyses for each patient across a time span
        for patient_id, patient_name in patients:
            print(f"Creating test data for patient {patient_name} (ID: {patient_id})")
            
            # Check how many analyses this patient already has
            cursor.execute("SELECT COUNT(*) FROM alzheimers_analysis WHERE patient_id = %s", (patient_id,))
            count = cursor.fetchone()[0]
            
            if count >= 3:
                print(f"  Patient already has {count} analyses - skipping")
                continue
            
            # Get a current timestamp
            now = datetime.now()
            
            # Create analyses at different dates
            # First visit (60 days ago - moderate condition)
            past_date1 = now - timedelta(days=60)
            
            # Generate score trend
            base_mmse = random.randint(19, 25)
            base_cdrsb = random.uniform(1.5, 4.0)
            
            # First visit scores (moderate)
            features1 = {
                "MMSE": base_mmse,
                "CDRSB": base_cdrsb,
                "ADAS11": random.uniform(12.0, 22.0),
                "ADAS13": random.uniform(18.0, 30.0),
                "RAVLT_immediate": random.uniform(20.0, 35.0),
                "RAVLT_learning": random.uniform(2.0, 5.0),
                "RAVLT_forgetting": random.uniform(3.0, 6.0),
                "Hippocampus": random.uniform(5500.0, 6500.0),
                "Entorhinal": random.uniform(3000.0, 3600.0),
                "Fusiform": random.uniform(16000.0, 18000.0),
                "MidTemp": random.uniform(18000.0, 20000.0),
                "Ventricles": random.uniform(20000.0, 25000.0),
                "WholeBrain": random.uniform(950000.0, 1050000.0),
                "ABETA": random.uniform(700.0, 1000.0),
                "TAU": random.uniform(250.0, 350.0),
                "PTAU": random.uniform(20.0, 40.0),
                "AGE": random.randint(65, 85),
                "APOE4": random.randint(0, 2)
            }
            
            # Second visit (30 days ago - slight improvement)
            past_date2 = now - timedelta(days=30)
            
            # Second visit scores (slight improvement)
            features2 = features1.copy()
            features2["MMSE"] = min(30, features1["MMSE"] + random.uniform(0.5, 2.0))  # Higher is better
            features2["CDRSB"] = max(0, features1["CDRSB"] - random.uniform(0.2, 0.8))  # Lower is better
            features2["ADAS11"] = max(0, features1["ADAS11"] - random.uniform(1.0, 3.0))  # Lower is better
            features2["ADAS13"] = max(0, features1["ADAS13"] - random.uniform(1.5, 4.0))  # Lower is better
            features2["RAVLT_immediate"] = min(75, features1["RAVLT_immediate"] + random.uniform(1.0, 4.0))  # Higher is better
            features2["Hippocampus"] = min(7500, features1["Hippocampus"] + random.uniform(100.0, 300.0))  # Higher is better
            
            # Third visit (recent - further improvement)
            past_date3 = now - timedelta(days=5)
            
            # Third visit scores (more improvement)
            features3 = features2.copy()
            features3["MMSE"] = min(30, features2["MMSE"] + random.uniform(0.5, 1.5))  # Higher is better
            features3["CDRSB"] = max(0, features2["CDRSB"] - random.uniform(0.1, 0.5))  # Lower is better
            features3["ADAS11"] = max(0, features2["ADAS11"] - random.uniform(0.5, 2.0))  # Lower is better
            features3["ADAS13"] = max(0, features2["ADAS13"] - random.uniform(1.0, 3.0))  # Lower is better
            features3["RAVLT_immediate"] = min(75, features2["RAVLT_immediate"] + random.uniform(1.0, 3.0))  # Higher is better
            features3["Hippocampus"] = min(7500, features2["Hippocampus"] + random.uniform(50.0, 200.0))  # Higher is better
            
            # Predict cognitive status based on MMSE score
            def predict_status(mmse):
                if mmse >= 27:
                    return "Nondemented", random.uniform(0.8, 0.95)
                elif mmse >= 24:
                    return "Converted", random.uniform(0.75, 0.9)
                else:
                    return "Demented", random.uniform(0.7, 0.9)
            
            # Create analyses with appropriate predictions
            prediction1, confidence1 = predict_status(features1["MMSE"])
            prediction2, confidence2 = predict_status(features2["MMSE"])
            prediction3, confidence3 = predict_status(features3["MMSE"])
            
            # Insert analyses
            insert_query = """
                INSERT INTO alzheimers_analysis
                (patient_id, input_features, prediction, confidence_score, analyzed_at)
                VALUES (%s, %s, %s, %s, %s)
            """
            
            # Insert only the analyses needed to get to 3 total
            analyses_to_add = 3 - count
            
            if analyses_to_add >= 1:
                cursor.execute(insert_query, (
                    patient_id, json.dumps(features1), prediction1, confidence1, past_date1
                ))
                print(f"  Added analysis 1: {prediction1} ({confidence1:.2f})")
            
            if analyses_to_add >= 2:
                cursor.execute(insert_query, (
                    patient_id, json.dumps(features2), prediction2, confidence2, past_date2
                ))
                print(f"  Added analysis 2: {prediction2} ({confidence2:.2f})")
            
            if analyses_to_add >= 3:
                cursor.execute(insert_query, (
                    patient_id, json.dumps(features3), prediction3, confidence3, past_date3
                ))
                print(f"  Added analysis 3: {prediction3} ({confidence3:.2f})")
            
            conn.commit()
    
    except Exception as e:
        print(f"Error creating test data: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    print("Creating test analysis data...")
    create_test_analyses()
    print("Done.") 