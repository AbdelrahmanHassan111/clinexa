import mysql.connector
import pandas as pd
from tabulate import tabulate
import os

# Database connection parameters - update these if needed
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "smart_clinic"
}

def print_divider(char="-", length=80):
    print(char * length)

def connect_to_db():
    """Connect to the database and return the connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except mysql.connector.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def print_schema():
    """Print the complete database schema"""
    print("\nDATABASE SCHEMA EXPLORER")
    print_divider("=")
    
    # Connect to the database
    conn = connect_to_db()
    if not conn:
        print("Failed to connect to the database. Check connection parameters.")
        return
    
    cursor = conn.cursor()
    
    # Get all tables
    try:
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in the database.")
            cursor.close()
            conn.close()
            return
        
        # Display the number of tables
        table_names = [table[0] for table in tables]
        print(f"Found {len(table_names)} tables in database '{DB_CONFIG['database']}'")
        print(", ".join(table_names))
        print_divider()
        
        # For each table, print its structure and some sample data
        for table_name in table_names:
            print(f"\nTABLE: {table_name}")
            print_divider("-")
            
            # Get table structure
            cursor.execute(f"DESCRIBE {table_name}")
            columns = cursor.fetchall()
            
            # Print table structure in tabular format
            column_data = []
            for col in columns:
                column_data.append([
                    col[0],  # Field
                    col[1],  # Type
                    "YES" if col[2] == "YES" else "NO",  # Null
                    col[3],  # Key
                    col[4] if col[4] is not None else "NULL",  # Default
                    col[5]   # Extra
                ])
            
            print("Table Structure:")
            headers = ["Field", "Type", "Nullable", "Key", "Default", "Extra"]
            print(tabulate(column_data, headers=headers, tablefmt="grid"))
            
            # Get foreign keys
            cursor.execute(f"""
            SELECT 
                COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
            FROM
                INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE
                REFERENCED_TABLE_NAME IS NOT NULL
                AND TABLE_NAME = %s
                AND TABLE_SCHEMA = %s
            """, (table_name, DB_CONFIG["database"]))
            
            foreign_keys = cursor.fetchall()
            
            if foreign_keys:
                print("\nForeign Keys:")
                fk_data = []
                for fk in foreign_keys:
                    fk_data.append([
                        fk[0],  # Column
                        f"{fk[1]}.{fk[2]}"  # Referenced Table.Column
                    ])
                print(tabulate(fk_data, headers=["Column", "References"], tablefmt="simple"))
            
            # Count rows in the table
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            print(f"\nTotal rows: {row_count}")
            
            # Show sample data if the table has rows
            if row_count > 0:
                limit = min(5, row_count)  # Show up to 5 rows
                cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
                sample_data = cursor.fetchall()
                
                # Get column names
                col_names = [desc[0] for desc in cursor.description]
                
                print(f"\nSample Data ({limit} rows):")
                sample_rows = []
                for row in sample_data:
                    # Truncate long values for better display
                    formatted_row = []
                    for val in row:
                        if isinstance(val, str) and len(val) > 50:
                            formatted_row.append(val[:47] + "...")
                        else:
                            formatted_row.append(val)
                    sample_rows.append(formatted_row)
                
                # Print sample data
                print(tabulate(sample_rows, headers=col_names, tablefmt="grid"))
            
            print_divider("=")
        
        # Print relationships between tables
        print("\nRELATIONSHIPS BETWEEN TABLES")
        print_divider("-")
        
        cursor.execute("""
        SELECT 
            TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
        FROM
            INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE
            REFERENCED_TABLE_NAME IS NOT NULL
            AND TABLE_SCHEMA = %s
        ORDER BY TABLE_NAME, REFERENCED_TABLE_NAME
        """, (DB_CONFIG["database"],))
        
        relationships = cursor.fetchall()
        
        if relationships:
            rel_data = []
            for rel in relationships:
                rel_data.append([
                    rel[0],  # Table
                    rel[1],  # Column
                    rel[2],  # Referenced Table
                    rel[3]   # Referenced Column
                ])
            
            print(tabulate(rel_data, headers=["Table", "Column", "References Table", "References Column"], tablefmt="grid"))
        else:
            print("No relationships (foreign keys) found between tables.")
        
    except mysql.connector.Error as e:
        print(f"Error exploring database: {e}")
    finally:
        # Close database connection
        cursor.close()
        conn.close()
        print("\nDatabase exploration complete.")

def log_to_file():
    """Run print_schema but log output to a file"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Redirect stdout to file
    import sys
    original_stdout = sys.stdout
    
    with open('logs/db_schema.txt', 'w') as f:
        sys.stdout = f
        print_schema()
        sys.stdout = original_stdout
    
    print(f"Schema output written to logs/db_schema.txt")

if __name__ == "__main__":
    # Ask the user whether to print to console or to a file
    print("Database Schema Explorer")
    print("1: Print schema to console")
    print("2: Save schema to file")
    choice = input("Choose option (1 or 2): ")
    
    if choice == "2":
        log_to_file()
    else:
        print_schema() 