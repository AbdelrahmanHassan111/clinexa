import mysql.connector
import streamlit as st
import os

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

def setup_mri_tables():
    """Set up MRI tables in the database."""
    # Connect to database
    conn = get_db_connection()
    if not conn:
        print("Could not connect to database")
        return False
    
    cursor = conn.cursor()
    
    try:
        # MRI Scans Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS mri_scans (
            scan_id INT AUTO_INCREMENT PRIMARY KEY,
            patient_id INT NOT NULL,
            scan_date DATETIME NOT NULL,
            scan_type VARCHAR(50) NOT NULL,
            file_path VARCHAR(255) NOT NULL,
            is_processed BOOLEAN DEFAULT FALSE,
            prediction VARCHAR(100) NULL,
            confidence FLOAT NULL,
            scan_notes TEXT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
        )
        """)
        
        # MRI ROI Measurements Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS mri_roi_measurements (
            measurement_id INT AUTO_INCREMENT PRIMARY KEY,
            scan_id INT NOT NULL,
            measurement_date DATETIME NOT NULL,
            hippocampus_left FLOAT NULL,
            hippocampus_right FLOAT NULL,
            hippocampus_total FLOAT NULL,
            entorhinal_left FLOAT NULL,
            entorhinal_right FLOAT NULL,
            entorhinal_total FLOAT NULL,
            lateral_ventricles FLOAT NULL,
            whole_brain FLOAT NULL,
            temporal_lobe_left FLOAT NULL,
            temporal_lobe_right FLOAT NULL,
            temporal_lobe_total FLOAT NULL,
            fusiform_left FLOAT NULL,
            fusiform_right FLOAT NULL,
            fusiform_total FLOAT NULL,
            amygdala_left FLOAT NULL,
            amygdala_right FLOAT NULL,
            amygdala_total FLOAT NULL,
            total_intracranial_volume FLOAT NULL,
            normalized_values JSON NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (scan_id) REFERENCES mri_scans(scan_id) ON DELETE CASCADE
        )
        """)
        
        # Create temp uploads directory if it doesn't exist
        os.makedirs("temp_uploads", exist_ok=True)
        os.makedirs("model", exist_ok=True)
        
        conn.commit()
        print("✅ MRI tables created successfully.")
        return True
    
    except mysql.connector.Error as e:
        print(f"Error setting up MRI tables: {e}")
        return False
    
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    setup_mri_tables() 