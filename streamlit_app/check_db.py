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
    
    # Check the structure of the role column in users table
    cursor.execute("DESCRIBE users role")
    role_structure = cursor.fetchone()
    print("Role column structure:", role_structure)
    
    # Check the values in the users table to see what roles exist
    cursor.execute("SELECT DISTINCT role FROM users")
    roles = cursor.fetchall()
    print("Existing roles in database:", roles)
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"Error: {e}") 