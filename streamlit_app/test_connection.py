import streamlit as st
import mysql.connector
import sys
import os
from pathlib import Path

st.set_page_config(page_title="Connection Test", layout="wide")

def main():
    st.title("🔌 Database Connection Test")
    st.write("This tool helps diagnose database connection issues during deployment.")
    
    # Get database configuration
    try:
        from db_config import DB_CONFIG
        connection_method = "Using db_config.py module"
    except ImportError:
        # Try multiple ways of getting configuration
        DB_CONFIG = None
        connection_method = "No configuration found"
        
        # Check Streamlit secrets
        if hasattr(st, "secrets") and "mysql" in st.secrets:
            DB_CONFIG = {
                "host": st.secrets.mysql.host,
                "user": st.secrets.mysql.user,
                "password": st.secrets.mysql.password,
                "database": st.secrets.mysql.database
            }
            connection_method = "Using Streamlit secrets"
        
        # Check environment variables
        elif all(os.environ.get(var) for var in ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]):
            DB_CONFIG = {
                "host": os.environ.get("DB_HOST"),
                "user": os.environ.get("DB_USER"),
                "password": os.environ.get("DB_PASSWORD"),
                "database": os.environ.get("DB_NAME")
            }
            connection_method = "Using environment variables"
        
        # Fallback to default
        else:
            DB_CONFIG = {
                "host": "localhost",
                "user": "root",
                "password": "root",
                "database": "smart_clinic"
            }
            connection_method = "Using default configuration"
    
    # Display configuration (hide password)
    st.subheader("Database Configuration")
    safe_config = DB_CONFIG.copy()
    if "password" in safe_config:
        safe_config["password"] = "********"
    
    st.json(safe_config)
    st.write(f"Configuration source: **{connection_method}**")
    
    # Test connection
    st.subheader("Connection Test")
    
    if st.button("Test Connection"):
        try:
            with st.spinner("Testing connection..."):
                # Try to connect
                connection = mysql.connector.connect(**DB_CONFIG)
                
                # Check if connected
                if connection.is_connected():
                    st.success("✅ Successfully connected to the database!")
                    
                    # Get server info
                    cursor = connection.cursor()
                    cursor.execute("SELECT VERSION()")
                    db_version = cursor.fetchone()[0]
                    
                    # Check required tables
                    cursor.execute("SHOW TABLES")
                    tables = [table[0] for table in cursor.fetchall()]
                    
                    required_tables = ["users", "patients", "doctors", "appointments", 
                                       "medical_records", "mri_scans", "alzheimers_analysis"]
                    
                    missing_tables = [table for table in required_tables if table not in tables]
                    
                    st.write(f"**MySQL Version:** {db_version}")
                    st.write(f"**Database:** {DB_CONFIG['database']}")
                    
                    if missing_tables:
                        st.warning(f"⚠️ Missing tables: {', '.join(missing_tables)}")
                        st.write("Consider running the schema update script.")
                    else:
                        st.write("All required tables exist in the database.")
                    
                    cursor.close()
                    connection.close()
                else:
                    st.error("❌ Connection test failed. Connected but is_connected() returned False.")
        
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
            
            # Provide debugging advice based on error
            if "Access denied" in str(e):
                st.info("This could be a username or password issue. Check your credentials.")
            elif "Can't connect to MySQL server" in str(e):
                st.info("Cannot reach the database server. Check the host address and that the server is running.")
            elif "Unknown database" in str(e):
                st.info(f"The database '{DB_CONFIG.get('database')}' does not exist. You may need to create it first.")
    
    # Additional deployment info
    st.subheader("Deployment Information")
    st.write(f"**Python version:** {sys.version}")
    st.write(f"**Running in Streamlit:** {st._is_running_with_streamlit}")
    
    # Check if running in Streamlit Cloud
    is_cloud = False
    if hasattr(st, "secrets") and "mysql" in st.secrets:
        is_cloud = True
    
    if is_cloud:
        st.write("**Environment:** Streamlit Cloud")
    else:
        st.write("**Environment:** Local deployment")
    
    # Help information
    with st.expander("Deployment Help"):
        st.markdown("""
        ### Common Issues and Solutions
        
        #### Database Connection Failures
        - **Wrong credentials**: Double-check username and password
        - **Server not reachable**: Ensure your database allows connections from the deployment server
        - **Missing database**: Create the 'smart_clinic' database if it doesn't exist
        - **Firewall issues**: Configure your database server to allow incoming connections
        
        #### For Streamlit Cloud Deployment
        1. Create a file `.streamlit/secrets.toml` with your database credentials
        2. Ensure your database is accessible from the internet
        3. Consider using a managed database service like AWS RDS
        
        #### For Local Deployment with ngrok
        1. Make sure your local MySQL server is running
        2. Default credentials should work if configured as in the code
        3. Run `python deploy_online.py` to start the ngrok tunnel
        """)

if __name__ == "__main__":
    main() 