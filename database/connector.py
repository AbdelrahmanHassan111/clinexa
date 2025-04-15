import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="smart_clinic"
)

cursor = conn.cursor()
cursor.execute("SELECT * FROM patients;")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
