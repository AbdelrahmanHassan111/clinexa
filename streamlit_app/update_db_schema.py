import mysql.connector

# Database connection parameters
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "smart_clinic"
}

try:
    # Connect to database
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Check the current structure of the role column
    cursor.execute("DESCRIBE users role")
    role_structure = cursor.fetchone()
    print("Current role column structure:", role_structure)
    
    # Check if the column is an ENUM and what values it allows
    if 'enum' in str(role_structure).lower():
        # Add 'patient' to the ENUM values if it's not already there
        cursor.execute("""
            ALTER TABLE users 
            MODIFY COLUMN role ENUM('admin', 'doctor', 'patient') NOT NULL
        """)
        print("Updated role column to include 'patient'")
    else:
        # If it's not an ENUM, print warning
        print("Warning: role column is not an ENUM type. Manual adjustment may be needed.")
    
    # Check the updated structure
    cursor.execute("DESCRIBE users role")
    updated_role = cursor.fetchone()
    print("Updated role column structure:", updated_role)
    
    conn.commit()
    cursor.close()
    conn.close()
    print("Database schema updated successfully")
    
except Exception as e:
    print(f"Error updating database schema: {e}") 