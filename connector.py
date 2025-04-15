import mysql.connector

def get_connection():
    """Create and return a connection to the MySQL database"""
    conn = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="root",
        database="smart_clinic"
    )
    return conn
