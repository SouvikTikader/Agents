import sqlite3

conn = sqlite3.connect("database/system.db")
conn.execute("DELETE FROM grievances;")
conn.commit()
conn.close()

print("✅ All grievances cleared successfully!")
