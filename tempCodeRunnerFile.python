import mysql.connector

def show_tables_and_columns():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",  # 🔁 replace with your MySQL password
        database="smart_clinic"        # 🔁 replace with your actual DB name
    )

    cursor = conn.cursor()

    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()

    print("\n📋 DATABASE STRUCTURE:\n")

    for table in tables:
        table_name = table[0]
        print(f"🔹 Table: {table_name}")
        cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        columns = cursor.fetchall()
        for col in columns:
            print(f"   - {col[0]} ({col[1]})")
        print("")

    cursor.close()
    conn.close()

# Run it
show_tables_and_columns()
