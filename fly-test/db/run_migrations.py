import dotenv
import psycopg
import os

def main(uri, migrations_dir):
    # print the database connection details
    print(f"Connecting to database at uri {uri}")

    # list files in migrations directory
    migration_files = sorted(os.listdir(migrations_dir))

    # print the list of migration files
    print("Migration files to be applied:")
    for migration_file in migration_files:
        print(migration_file)

    migrations = []
    for migration_file in migration_files:
        with open(os.path.join(migrations_dir, migration_file), "r") as f:
            migrations.append(f.read())

    conn = psycopg.connect(uri)

    with conn.cursor() as cur:

        for migration in migrations:
            cur.execute(migration)

        conn.commit()

        # print "SUCCESS", and then db summary
        print("SUCCESS")

        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        tables = cur.fetchall()
        print("Tables in the database:")
        for table in tables:
            print(table[0])

        # print table information (columns, num rows)
        for table in tables:
            print(f"Information for table {table[0]}:")
            cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{table[0]}'")
            columns = cur.fetchall()
            print("Columns:")
            for column in columns:
                print(f"  {column[0]} ({column[1]})")
            cur.execute(f"SELECT COUNT(*) FROM {table[0]}")
            num_rows = cur.fetchone()[0]
            print(f"Number of rows: {num_rows}")


if __name__ == "__main__":
    dotenv.load_dotenv()

    main(
        uri=os.getenv("DB_URI"),
        migrations_dir="migrations/",
    )