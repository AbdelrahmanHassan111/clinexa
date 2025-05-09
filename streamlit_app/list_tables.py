import mysql.connector

# Database connection parameters - same as in the app
try:
    from db_config import DB_CONFIG
except ImportError:
    # Fallback configuration if db_config.py is not available
    DB_CONFIG = {
        "host": "localhost",
        "user": "root",
        "password": "root",
        "database": "smart_clinic"
    }

def connect_to_database():
    """Create a connection to the MySQL database"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        print("✅ Successfully connected to database")
        return connection
    except mysql.connector.Error as e:
        print(f"❌ Database connection error: {e}")
        return None

def main():
    """Simple script to list database tables and structure"""
    conn = connect_to_database()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    
    print(f"\nDatabase contains {len(tables)} tables:")
    for i, table in enumerate(tables, 1):
        table_name = table[0]
        print(f"{i}. {table_name}")
        
        # Get table structure
        cursor.execute(f"DESCRIBE {table_name}")
        columns = cursor.fetchall()
        print(f"   Columns ({len(columns)}):")
        for col in columns:
            print(f"     - {col[0]} ({col[1]})")
    
    # Check for MRI-related tables
    print("\nSearching for MRI-related tables...")
    mri_tables = [table[0] for table in tables if 'mri' in table[0].lower()]
    if mri_tables:
        print(f"Found {len(mri_tables)} MRI-related tables:")
        for table in mri_tables:
            print(f"  - {table}")
            
            # Get detailed structure of MRI tables
            cursor.execute(f"DESCRIBE {table}")
            columns = cursor.fetchall()
            print(f"    Columns:")
            for col in columns:
                print(f"      - {col[0]} ({col[1]})")
    else:
        print("No MRI-related tables found.")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main() 