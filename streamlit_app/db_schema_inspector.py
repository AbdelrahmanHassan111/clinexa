import mysql.connector
import pandas as pd
from tabulate import tabulate
import sys
import os

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
        sys.exit(1)

def get_tables(connection):
    """Get all tables in the database"""
    cursor = connection.cursor()
    cursor.execute("SHOW TABLES")
    tables = [table[0] for table in cursor.fetchall()]
    cursor.close()
    return tables

def get_table_schema(connection, table_name):
    """Get detailed schema for a specific table"""
    cursor = connection.cursor(dictionary=True)
    
    # Get column information
    cursor.execute(f"DESCRIBE {table_name}")
    columns = cursor.fetchall()
    
    # Get index information
    cursor.execute(f"SHOW INDEX FROM {table_name}")
    indexes = cursor.fetchall()
    
    # Get foreign key information if supported
    foreign_keys = []
    try:
        cursor.execute(f"""
            SELECT 
                TABLE_NAME, COLUMN_NAME, CONSTRAINT_NAME, 
                REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
            FROM
                INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE
                REFERENCED_TABLE_SCHEMA = '{DB_CONFIG['database']}'
                AND TABLE_NAME = '{table_name}'
        """)
        foreign_keys = cursor.fetchall()
    except:
        # MySql might not support this query in all versions
        pass
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
    row_count = cursor.fetchone()['count']
    
    # Get sample data (first 3 rows)
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
    sample_data = cursor.fetchall()
    
    cursor.close()
    return {
        "columns": columns, 
        "indexes": indexes, 
        "foreign_keys": foreign_keys,
        "row_count": row_count,
        "sample_data": sample_data
    }

def save_schema_to_file(output):
    """Save the schema output to a file for later reference"""
    output_dir = "database_docs"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "database_schema.txt")
    with open(output_file, "w") as f:
        f.write(output)
    print(f"\nSchema documentation saved to {output_file}")

def main():
    """Main function to inspect and display database schema"""
    connection = connect_to_database()
    tables = get_tables(connection)
    
    # Collect output in a string for both display and file saving
    output = []
    output.append(f"\n🗂️ Database '{DB_CONFIG['database']}' contains {len(tables)} tables:")
    for i, table in enumerate(sorted(tables), 1):
        output.append(f"{i}. {table}")
    
    output.append("\n📊 Detailed Schema for each table:")
    for table in sorted(tables):
        schema = get_table_schema(connection, table)
        
        output.append(f"\n{'='*80}")
        output.append(f"📋 TABLE: {table} ({schema['row_count']} rows)")
        output.append(f"{'='*80}")
        
        # Show columns
        output.append("\n📌 COLUMNS:")
        columns_df = pd.DataFrame(schema["columns"])
        output.append(tabulate(columns_df, headers="keys", tablefmt="psql"))
        
        # Show indexes
        if schema["indexes"]:
            output.append("\n🔑 INDEXES:")
            indexes_df = pd.DataFrame(schema["indexes"])
            # Select most relevant columns for display
            if "Key_name" in indexes_df.columns and "Column_name" in indexes_df.columns:
                indexes_df = indexes_df[["Key_name", "Column_name", "Non_unique", "Seq_in_index"]]
            output.append(tabulate(indexes_df, headers="keys", tablefmt="psql"))
        
        # Show foreign keys
        if schema["foreign_keys"]:
            output.append("\n🔗 FOREIGN KEYS:")
            fk_df = pd.DataFrame(schema["foreign_keys"])
            output.append(tabulate(fk_df, headers="keys", tablefmt="psql"))
        
        # Show sample data
        if schema["sample_data"]:
            output.append(f"\n📝 SAMPLE DATA (first 3 rows):")
            sample_df = pd.DataFrame(schema["sample_data"])
            try:
                if not sample_df.empty:
                    output.append(tabulate(sample_df, headers="keys", tablefmt="psql", maxcolwidths=30))
                else:
                    output.append("No sample data available (table is empty)")
            except Exception as e:
                output.append(f"Error displaying sample data: {e}")
    
    # Display all relationships (table diagram)
    output.append("\n\n🔄 DATABASE RELATIONSHIPS:")
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                TABLE_NAME as source_table, 
                COLUMN_NAME as source_column,
                REFERENCED_TABLE_NAME as target_table,
                REFERENCED_COLUMN_NAME as target_column
            FROM
                INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE
                REFERENCED_TABLE_SCHEMA = %s
            ORDER BY
                source_table, target_table
        """, (DB_CONFIG['database'],))
        relationships = cursor.fetchall()
        cursor.close()
        
        if relationships:
            rel_df = pd.DataFrame(relationships)
            output.append(tabulate(rel_df, headers="keys", tablefmt="psql"))
            
            # Generate a simple ASCII diagram of relationships
            output.append("\n📊 RELATIONSHIP DIAGRAM:")
            for rel in relationships:
                output.append(f"  {rel['source_table']}.{rel['source_column']} ──> {rel['target_table']}.{rel['target_column']}")
        else:
            output.append("No explicit relationships found in database schema.")
    except Exception as e:
        output.append(f"Could not extract relationships: {e}")
    
    connection.close()
    output.append("\n✅ Database schema inspection complete")
    
    # Convert output list to string
    output_str = "\n".join(output)
    
    # Print to console
    print(output_str)
    
    # Save to file
    save_schema_to_file(output_str)
    
    # Look specifically for MRI-related tables
    mri_tables = [table for table in tables if 'mri' in table.lower()]
    if mri_tables:
        print("\n🧠 MRI-SPECIFIC TABLES FOUND:")
        for table in mri_tables:
            print(f"  - {table}")
    else:
        print("\n⚠️ No MRI-specific tables found in the database schema.")

if __name__ == "__main__":
    main() 