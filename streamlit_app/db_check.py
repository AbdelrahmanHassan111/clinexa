import mysql.connector
import streamlit as st

def check_db_tables():
    # Database connection parameters
    DB_CONFIG = {
        "host": "localhost",
        "user": "root", 
        "password": "root",
        "database": "smart_clinic"
    }
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Get all tables in the database
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print("Tables in database:")
        for table in tables:
            print(f"- {table[0]}")
            
        # Check specific tables needed for the application
        required_tables = [
            "alzheimers_analysis",
            "medical_records",
            "patients",
            "doctors",
            "users",
            "mri_scans"
        ]
        
        missing_tables = []
        for table in required_tables:
            cursor.execute(f"SHOW TABLES LIKE '{table}'")
            result = cursor.fetchall()
            if not result:
                missing_tables.append(table)
        
        if missing_tables:
            print(f"Missing tables: {', '.join(missing_tables)}")
        
        # Check alzheimers_analysis table schema if it exists
        cursor.execute("SHOW TABLES LIKE 'alzheimers_analysis'")
        if cursor.fetchall():
            print("\nAlzheimers Analysis table schema:")
            cursor.execute("DESCRIBE alzheimers_analysis")
            columns = cursor.fetchall()
            for col in columns:
                print(f"- {col[0]}: {col[1]}")
                
            # Check sample data in the table
            print("\nSample data in alzheimers_analysis:")
            cursor.execute("SELECT analysis_id, patient_id, prediction, confidence_score, analyzed_at FROM alzheimers_analysis LIMIT 5")
            sample_data = cursor.fetchall()
            if sample_data:
                for row in sample_data:
                    print(f"ID: {row[0]}, Patient: {row[1]}, Prediction: {row[2]}, Confidence: {row[3]}, Date: {row[4]}")
            else:
                print("No data found in alzheimers_analysis table")
                
            # Check input_features column specifically
            print("\nChecking input_features column:")
            cursor.execute("SELECT analysis_id, input_features FROM alzheimers_analysis LIMIT 2")
            features_data = cursor.fetchall()
            if features_data:
                for row in features_data:
                    print(f"ID: {row[0]}")
                    if row[1]:
                        print(f"Features present: {'Yes' if len(row[1]) > 10 else 'No'}")
                        print(f"Features length: {len(row[1])}")
                        print(f"Sample features: {row[1][:200]}...")
                    else:
                        print("Features: None")
            else:
                print("No input_features data found")
        
        cursor.close()
        conn.close()
        
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")

if __name__ == "__main__":
    check_db_tables() 