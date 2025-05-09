import mysql.connector
import pandas as pd
from tabulate import tabulate
import os
import sys

# Database connection parameters - update these if needed
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "smart_clinic"
}

def print_divider(char="-", length=80):
    print(char * length)

def get_schema():
    """Generate the database schema information as a string"""
    output = []
    output.append("\nDATABASE SCHEMA EXPLORER")
    output.append("=" * 80)
    
    # Connect to the database
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        output.append(f"Error connecting to database: {e}")
        return "\n".join(output)
    
    cursor = connection.cursor()
    
    # Get all tables
    try:
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        if not tables:
            output.append("No tables found in the database.")
            cursor.close()
            connection.close()
            return "\n".join(output)
        
        # Display the number of tables
        table_names = [table[0] for table in tables]
        output.append(f"Found {len(table_names)} tables in database '{DB_CONFIG['database']}'")
        output.append(", ".join(table_names))
        output.append("-" * 80)
        
        # For each table, get its structure and some sample data
        for table_name in table_names:
            output.append(f"\nTABLE: {table_name}")
            output.append("-" * 80)
            
            # Get table structure
            cursor.execute(f"DESCRIBE {table_name}")
            columns = cursor.fetchall()
            
            # Format table structure
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
            
            output.append("Table Structure:")
            headers = ["Field", "Type", "Nullable", "Key", "Default", "Extra"]
            output.append(tabulate(column_data, headers=headers, tablefmt="grid"))
            
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
                output.append("\nForeign Keys:")
                fk_data = []
                for fk in foreign_keys:
                    fk_data.append([
                        fk[0],  # Column
                        f"{fk[1]}.{fk[2]}"  # Referenced Table.Column
                    ])
                output.append(tabulate(fk_data, headers=["Column", "References"], tablefmt="simple"))
            
            # Count rows in the table
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            output.append(f"\nTotal rows: {row_count}")
            
            # Show sample data if the table has rows
            if row_count > 0:
                limit = min(5, row_count)  # Show up to 5 rows
                cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
                sample_data = cursor.fetchall()
                
                # Get column names
                col_names = [desc[0] for desc in cursor.description]
                
                output.append(f"\nSample Data ({limit} rows):")
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
                
                # Format sample data
                output.append(tabulate(sample_rows, headers=col_names, tablefmt="grid"))
            
            output.append("=" * 80)
        
        # Get relationships between tables
        output.append("\nRELATIONSHIPS BETWEEN TABLES")
        output.append("-" * 80)
        
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
            
            output.append(tabulate(rel_data, headers=["Table", "Column", "References Table", "References Column"], tablefmt="grid"))
        else:
            output.append("No relationships (foreign keys) found between tables.")
        
        output.append("\nDatabase exploration complete.")
        
    except mysql.connector.Error as e:
        output.append(f"Error exploring database: {e}")
    finally:
        # Close database connection
        cursor.close()
        connection.close()
    
    return "\n".join(output)

def main():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Get the schema output
    schema_text = get_schema()
    
    # Save to file
    filename = 'logs/db_schema.txt'
    with open(filename, 'w') as f:
        f.write(schema_text)
    
    print(f"Database schema saved to {filename}")
    
    # Also print to console
    print("\nSummary of database tables:")
    
    # Try to connect to the database to get a quick summary
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Get tables and row counts
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        table_data = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            table_data.append([table_name, row_count])
        
        print(tabulate(table_data, headers=["Table Name", "Row Count"], tablefmt="simple"))
        cursor.close()
        conn.close()
        
    except mysql.connector.Error as e:
        print(f"Error creating summary: {e}")

if __name__ == "__main__":
    main() 