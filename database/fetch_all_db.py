import mysql.connector

def describe_database_schema():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",  # 🔁 replace with your MySQL password
        database="smart_clinic"
    )

    cursor = conn.cursor(dictionary=True)

    # Fetch all tables
    cursor.execute("""
        SELECT TABLE_NAME, TABLE_TYPE, ENGINE, TABLE_ROWS
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = DATABASE()
    """)
    tables = cursor.fetchall()

    print("\n📚 FULL DATABASE SCHEMA:\n")

    for table in tables:
        table_name = table['TABLE_NAME']
        table_type = table['TABLE_TYPE']
        engine = table['ENGINE']
        rows = table['TABLE_ROWS']

        print(f"🔹 {table_type}: {table_name} (Engine: {engine}, Rows: {rows})")

        # Columns
        cursor.execute("""
            SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT, EXTRA, COLUMN_KEY
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
        """, (table_name,))
        columns = cursor.fetchall()

        for col in columns:
            print(f"   - Column: {col['COLUMN_NAME']}")
            print(f"     ▸ Type: {col['COLUMN_TYPE']}")
            print(f"     ▸ Nullable: {col['IS_NULLABLE']}")
            print(f"     ▸ Default: {col['COLUMN_DEFAULT']}")
            print(f"     ▸ Extra: {col['EXTRA']}")
            print(f"     ▸ Key: {col['COLUMN_KEY']}")

        # Constraints
        cursor.execute("""
            SELECT CONSTRAINT_NAME, CONSTRAINT_TYPE
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
        """, (table_name,))
        constraints = cursor.fetchall()
        if constraints:
            print("   📎 Constraints:")
            for c in constraints:
                print(f"     - {c['CONSTRAINT_TYPE']}: {c['CONSTRAINT_NAME']}")

        # Foreign keys
        cursor.execute("""
            SELECT
                kcu.CONSTRAINT_NAME, kcu.COLUMN_NAME,
                kcu.REFERENCED_TABLE_NAME, kcu.REFERENCED_COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
            JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
              ON kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
              AND kcu.TABLE_SCHEMA = tc.TABLE_SCHEMA
              AND kcu.TABLE_NAME = tc.TABLE_NAME
            WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
              AND kcu.TABLE_SCHEMA = DATABASE()
              AND kcu.TABLE_NAME = %s
        """, (table_name,))
        fks = cursor.fetchall()
        if fks:
            print("   🔗 Foreign Keys:")
            for fk in fks:
                print(f"     - {fk['COLUMN_NAME']} → {fk['REFERENCED_TABLE_NAME']}.{fk['REFERENCED_COLUMN_NAME']}")

        # Indexes
        cursor.execute("""
            SHOW INDEXES FROM `{}` FROM `{}` 
        """.format(table_name, conn.database))
        indexes = cursor.fetchall()
        if indexes:
            print("   🧩 Indexes:")
            for idx in indexes:
                print(f"     - {idx['Key_name']} on {idx['Column_name']} (Unique: {'No' if idx['Non_unique'] else 'Yes'})")

        print("")

    cursor.close()
    conn.close()

# Run it
describe_database_schema()
