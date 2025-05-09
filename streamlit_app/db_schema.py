import streamlit as st
import mysql.connector
import pandas as pd

# Set page config
st.set_page_config(page_title="Database Schema Explorer", layout="wide")

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
        st.error(f"Database connection error: {e}")
        return None

def main():
    st.title("🔍 Database Schema Explorer")
    st.markdown("This tool helps understand the database structure for troubleshooting purposes.")
    
    # Try to connect to the database
    conn = get_db_connection()
    
    if not conn:
        st.error("❌ Could not connect to the database. Please check your connection parameters.")
        
        # Show form to update connection parameters
        with st.form("db_connection_form"):
            st.subheader("Update Database Connection Parameters")
            host = st.text_input("Host", value=DB_CONFIG["host"])
            user = st.text_input("Username", value=DB_CONFIG["user"])
            password = st.text_input("Password", value=DB_CONFIG["password"], type="password")
            database = st.text_input("Database Name", value=DB_CONFIG["database"])
            
            if st.form_submit_button("Test Connection"):
                test_config = {
                    "host": host,
                    "user": user,
                    "password": password,
                    "database": database
                }
                try:
                    test_conn = mysql.connector.connect(**test_config)
                    st.success("✅ Connection successful!")
                    test_conn.close()
                except Exception as e:
                    st.error(f"Connection failed: {e}")
    else:
        st.success("✅ Connected to the database successfully!")
        
        # Get all tables in the database
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        if not tables:
            st.warning("No tables found in the database.")
        else:
            # Display all tables in a selectbox
            table_names = [table[0] for table in tables]
            st.subheader("📊 Database Tables")
            
            # Create a multi-column layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_table = st.selectbox("Select a table to view its schema", table_names)
            
            # Button to view all tables
            with col2:
                view_all = st.checkbox("View all tables at once", value=False)
            
            if view_all:
                # Display schema for all tables
                for table_name in table_names:
                    st.markdown(f"### 📋 Table: `{table_name}`")
                    
                    # Get table schema
                    cursor.execute(f"DESCRIBE {table_name}")
                    columns = cursor.fetchall()
                    
                    # Convert to dataframe for better display
                    df_schema = pd.DataFrame(columns, columns=["Field", "Type", "Null", "Key", "Default", "Extra"])
                    st.dataframe(df_schema)
                    
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]
                    st.info(f"Total rows: {row_count}")
                    
                    # Preview data if table has rows
                    if row_count > 0:
                        with st.expander(f"Preview data for {table_name}"):
                            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
                            data = cursor.fetchall()
                            # Get column names
                            col_names = [i[0] for i in cursor.description]
                            df_data = pd.DataFrame(data, columns=col_names)
                            st.dataframe(df_data)
                    
                    st.markdown("---")
            else:
                # Display schema for selected table
                if selected_table:
                    st.markdown(f"### 📋 Table: `{selected_table}`")
                    
                    # Get table schema
                    cursor.execute(f"DESCRIBE {selected_table}")
                    columns = cursor.fetchall()
                    
                    # Convert to dataframe for better display
                    df_schema = pd.DataFrame(columns, columns=["Field", "Type", "Null", "Key", "Default", "Extra"])
                    st.dataframe(df_schema)
                    
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {selected_table}")
                    row_count = cursor.fetchone()[0]
                    st.info(f"Total rows: {row_count}")
                    
                    # Show a preview of the data
                    if row_count > 0:
                        st.subheader("Data Preview")
                        cursor.execute(f"SELECT * FROM {selected_table} LIMIT 10")
                        data = cursor.fetchall()
                        # Get column names
                        col_names = [i[0] for i in cursor.description]
                        df_data = pd.DataFrame(data, columns=col_names)
                        st.dataframe(df_data)
                    
                    # Show any foreign keys
                    cursor.execute(f"""
                    SELECT 
                        TABLE_NAME, COLUMN_NAME, CONSTRAINT_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                    FROM
                        INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE
                        REFERENCED_TABLE_NAME IS NOT NULL
                        AND TABLE_NAME = %s
                    """, (selected_table,))
                    
                    foreign_keys = cursor.fetchall()
                    if foreign_keys:
                        st.subheader("Foreign Keys")
                        df_fk = pd.DataFrame(foreign_keys, columns=[
                            "Table", "Column", "Constraint Name", "Referenced Table", "Referenced Column"
                        ])
                        st.dataframe(df_fk)
            
            # Show ER diagram button
            st.subheader("Database Relationships")
            st.info("Below are the relationships between tables based on foreign keys")
            
            cursor.execute("""
            SELECT 
                TABLE_NAME, COLUMN_NAME, CONSTRAINT_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
            FROM
                INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE
                REFERENCED_TABLE_NAME IS NOT NULL
                AND TABLE_SCHEMA = %s
            """, (DB_CONFIG["database"],))
            
            relationships = cursor.fetchall()
            if relationships:
                df_relationships = pd.DataFrame(relationships, columns=[
                    "Table", "Column", "Constraint Name", "Referenced Table", "Referenced Column"
                ])
                st.dataframe(df_relationships)
                
                # Display a simple visualization of relationships
                st.markdown("### 🔄 Relationship Visualization")
                st.markdown("Simple visualization of table relationships:")
                
                # Generate a text-based representation of relationships
                relation_text = ""
                for _, row in df_relationships.iterrows():
                    relation_text += f"{row['Table']} ({row['Column']}) → {row['Referenced Table']} ({row['Referenced Column']})\n"
                
                st.code(relation_text)
            else:
                st.warning("No relationships (foreign keys) found between tables.")
        
        # Close database connection
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main() 