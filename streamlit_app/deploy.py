import streamlit as st
import mysql.connector
import os
import subprocess
import sys

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

def update_database_schema():
    """Update the database schema with new tables and fields."""
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database. Please check your credentials.")
        return False
    
    cursor = conn.cursor()
    
    try:
        # Update users table
        cursor.execute("""
            ALTER TABLE users 
            ADD COLUMN patient_id INT NULL
        """)
        print("Added patient_id column to users table")
    except Exception as e:
        print(f"Note: {e}")  # This might already exist
    
    try:
        cursor.execute("""
            ALTER TABLE users 
            ADD CONSTRAINT fk_user_patient FOREIGN KEY (patient_id) 
            REFERENCES patients(patient_id)
        """)
        print("Added foreign key constraint to users table")
    except Exception as e:
        print(f"Note: {e}")  # This might already exist
    
    # Update patients table
    try:
        cursor.execute("""
            ALTER TABLE patients 
            ADD COLUMN email VARCHAR(100) NULL
        """)
        print("Added email column to patients table")
    except Exception as e:
        print(f"Note: {e}")  # This might already exist
    
    try:
        cursor.execute("""
            ALTER TABLE patients 
            ADD COLUMN emergency_contact VARCHAR(100) NULL
        """)
        print("Added emergency_contact column to patients table")
    except Exception as e:
        print(f"Note: {e}")  # This might already exist
    
    try:
        cursor.execute("""
            ALTER TABLE patients 
            ADD COLUMN emergency_phone VARCHAR(20) NULL
        """)
        print("Added emergency_phone column to patients table")
    except Exception as e:
        print(f"Note: {e}")  # This might already exist
    
    try:
        cursor.execute("""
            ALTER TABLE patients 
            ADD COLUMN allergies TEXT NULL
        """)
        print("Added allergies column to patients table")
    except Exception as e:
        print(f"Note: {e}")  # This might already exist
    
    try:
        cursor.execute("""
            ALTER TABLE patients 
            ADD COLUMN medical_conditions TEXT NULL
        """)
        print("Added medical_conditions column to patients table")
    except Exception as e:
        print(f"Note: {e}")  # This might already exist
    
    try:
        cursor.execute("""
            ALTER TABLE patients 
            ADD CONSTRAINT unique_patient_email UNIQUE (email)
        """)
        print("Added unique constraint on email column")
    except Exception as e:
        print(f"Note: {e}")  # This might already exist
    
    # Update appointments table
    try:
        cursor.execute("""
            ALTER TABLE appointments 
            ADD COLUMN created_at DATETIME NULL
        """)
        print("Added created_at column to appointments table")
    except Exception as e:
        print(f"Note: {e}")  # This might already exist
    
    try:
        cursor.execute("""
            ALTER TABLE appointments 
            ADD COLUMN cancellation_date DATETIME NULL
        """)
        print("Added cancellation_date column to appointments table")
    except Exception as e:
        print(f"Note: {e}")  # This might already exist
    
    cursor.close()
    conn.close()
    print("Database schema update completed successfully")
    return True

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        "streamlit", 
        "mysql-connector-python", 
        "pandas", 
        "numpy",
        "matplotlib",
        "google-generativeai",
        "python-dotenv",
        "joblib",
        "scikit-learn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        
        # Ask if user wants to install missing packages
        answer = input("Do you want to install missing packages? (y/n): ")
        if answer.lower() == 'y':
            for package in missing_packages:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print("All required packages are now installed")
            return True
        else:
            print("Please install missing packages manually and try again")
            return False
    
    return True

def start_application():
    """Start the Streamlit application."""
    print("Starting Alzheimer Diagnosis System...")
    subprocess.Popen(["streamlit", "run", "app.py"])
    print("Application started! You can access it in your browser.")

def deploy_system():
    """Deploy the full system."""
    print("=" * 50)
    print("Alzheimer Diagnosis System - Deployment Script")
    print("=" * 50)
    
    # Check dependencies
    print("\nChecking required dependencies...")
    if not check_dependencies():
        return
    
    # Update database schema
    print("\nUpdating database schema...")
    if not update_database_schema():
        return
    
    # Start application
    print("\nStarting application...")
    start_application()
    
    print("\nDeployment completed successfully!")
    print("You can access the application at http://localhost:8501")

if __name__ == "__main__":
    deploy_system() 