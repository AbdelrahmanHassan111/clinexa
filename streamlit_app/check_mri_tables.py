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
    """Detailed inspection of MRI-related tables"""
    conn = connect_to_database()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    # Get list of MRI-related tables
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    mri_tables = [table[0] for table in tables if 'mri' in table[0].lower()]
    
    if not mri_tables:
        print("No MRI-related tables found in the database.")
        return
    
    print(f"\nFound {len(mri_tables)} MRI-related tables:")
    for table in mri_tables:
        print(f"\n{'='*40}")
        print(f"TABLE: {table}")
        print(f"{'='*40}")
        
        # Get table structure
        cursor.execute(f"DESCRIBE {table}")
        columns = cursor.fetchall()
        print(f"Columns:")
        for col in columns:
            field_name = col[0]
            field_type = col[1]
            nullable = "NULL" if col[2] == "YES" else "NOT NULL"
            key = col[3] if col[3] else "---"
            default = col[4] if col[4] else "---"
            extra = col[5] if col[5] else "---"
            
            print(f"  - {field_name}: {field_type}, {nullable}, Key: {key}, Default: {default}, {extra}")
        
        # Get foreign keys
        try:
            cursor.execute(f"""
                SELECT 
                    COLUMN_NAME, CONSTRAINT_NAME, 
                    REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                FROM
                    INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE
                    REFERENCED_TABLE_SCHEMA = '{DB_CONFIG['database']}'
                    AND TABLE_NAME = '{table}'
            """)
            foreign_keys = cursor.fetchall()
            
            if foreign_keys:
                print(f"\nForeign Keys:")
                for fk in foreign_keys:
                    print(f"  - {fk[0]} → {fk[2]}.{fk[3]} (Constraint: {fk[1]})")
        except Exception as e:
            print(f"Error fetching foreign keys: {e}")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = cursor.fetchone()[0]
        print(f"\nRow Count: {row_count}")
        
        # Get sample data
        if row_count > 0:
            try:
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                sample_data = cursor.fetchall()
                
                if sample_data:
                    print(f"\nSample Data (up to 3 rows):")
                    for i, row in enumerate(sample_data, 1):
                        print(f"  Row {i}:")
                        for j, col in enumerate(row):
                            col_name = columns[j][0] if j < len(columns) else f"Column{j}"
                            print(f"    {col_name}: {str(col)[:100]}")
            except Exception as e:
                print(f"Error fetching sample data: {e}")
    
    # Check for references to MRI tables
    print("\n\nReferences to MRI tables:")
    for mri_table in mri_tables:
        try:
            cursor.execute(f"""
                SELECT 
                    TABLE_NAME, COLUMN_NAME, CONSTRAINT_NAME
                FROM
                    INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE
                    REFERENCED_TABLE_SCHEMA = '{DB_CONFIG['database']}'
                    AND REFERENCED_TABLE_NAME = '{mri_table}'
            """)
            references = cursor.fetchall()
            
            if references:
                print(f"\nReferences to {mri_table}:")
                for ref in references:
                    print(f"  - Table: {ref[0]}, Column: {ref[1]}, Constraint: {ref[2]}")
            else:
                print(f"No tables reference {mri_table}")
        except Exception as e:
            print(f"Error checking references to {mri_table}: {e}")
    
    # Check specifically for the missing mri_scans table
    print("\n\nChecking for mri_scans table...")
    cursor.execute("SHOW TABLES LIKE 'mri_scans'")
    if cursor.fetchone():
        print("✅ mri_scans table exists")
        
        # Get detailed info about mri_scans
        cursor.execute("DESCRIBE mri_scans")
        columns = cursor.fetchall()
        print("Columns:")
        for col in columns:
            print(f"  - {col[0]} ({col[1]})")
            
        # Get row count
        cursor.execute("SELECT COUNT(*) FROM mri_scans")
        count = cursor.fetchone()[0]
        print(f"Row count: {count}")
    else:
        print("❌ mri_scans table does not exist")
        print("This is a critical missing table. The data model requires an mri_scans table!")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main() 