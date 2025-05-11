import mysql.connector
import sys

# Database connection parameters
DB_CONFIG = {
    "host": "clinexa.cgpek8igovya.us-east-1.rds.amazonaws.com",
    "port": 3306,
    "user": "clinexa",
    "password": "Am24268934",
    "database": "clinexa_db"
}

def get_db_connection():
    """Create a direct database connection."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def explore_database():
    """Explore the database structure"""
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database")
        return
    
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    
    print("Tables in database:")
    for table in tables:
        table_name = table[0]
        print(f"\n--- Table: {table_name} ---")
        
        # Get table columns
        cursor.execute(f"DESCRIBE {table_name}")
        columns = cursor.fetchall()
        print("Columns:")
        for col in columns:
            print(f"  {col[0]} - {col[1]}")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        print(f"Row count: {row_count}")
        
        # If the table is 'appointments', get column names and first row
        if table_name == 'appointments':
            print("\nAppointments table details:")
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
            column_names = [desc[0] for desc in cursor.description]
            print("Column names:", column_names)
            
            row = cursor.fetchone()
            if row:
                print("Sample row:")
                for i, value in enumerate(row):
                    print(f"  {column_names[i]}: {value}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    explore_database() 