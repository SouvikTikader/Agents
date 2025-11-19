from werkzeug.security import check_password_hash
import sqlite3

conn = sqlite3.connect('database/system.db')
conn.row_factory = sqlite3.Row
user = conn.execute("SELECT * FROM users WHERE username=?", ("ADT23SOCB0100",)).fetchone()



# ---------------------------------------------------------
# Sample inserts for users and performance (ensure users exist)
# ---------------------------------------------------------

# Insert sample users (students)
sample_students = [
    ('ADT23SOCB0001', 'student1@example.com', 'hashedpassword1', 'student'),
    ('ADT23SOCB0002', 'student2@example.com', 'hashedpassword2', 'student'),
    ('ADT23SOCB0003', 'student3@example.com', 'hashedpassword3', 'student'),
    ('ADT23SOCB0004', 'student4@example.com', 'hashedpassword4', 'student'),
    ('ADT23SOCB0005', 'student5@example.com', 'hashedpassword5', 'student'),
    ('ADT23SOCB0006', 'student6@example.com', 'hashedpassword6', 'student'),
    ('ADT23SOCB0007', 'student7@example.com', 'hashedpassword7', 'student'),
    ('ADT23SOCB0008', 'student8@example.com', 'hashedpassword8', 'student'),
    ('ADT23SOCB0009', 'student9@example.com', 'hashedpassword9', 'student'),
    ('ADT23SOCB0010', 'student10@example.com', 'hashedpassword10', 'student'),
    ('ADT23SOCB0011', 'student11@example.com', 'hashedpassword11', 'student'),
]

conn.executemany('''
    INSERT OR IGNORE INTO users (username, email, password, role)
    VALUES (?, ?, ?, ?)
''', sample_students)

# Insert sample performance data
conn.executemany('''
    INSERT OR IGNORE INTO performance (
        student_id, attendance_before_ta1, marks_ta1,
        attendance_before_ta2, marks_ta2,
        attendance_before_final, assignment_marks
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
''', [
    ('ADT23SOCB0001', 85.5, 90.0, 88.0, 92.0, 90.0, 85.0),
    ('ADT23SOCB0002', 78.0, 88.5, 80.0, 85.0, 75.0, 80.0),
    ('ADT23SOCB0003', 92.0, 95.0, 90.0, 94.0, 96.0, 95.0),
    ('ADT23SOCB0004', 60.0, 70.0, 65.0, 68.0, 70.0, 60.0),
    ('ADT23SOCB0005', 88.0, 82.0, 85.0, 80.0, 87.0, 82.0),
    ('ADT23SOCB0006', 74.5, 76.0, 72.0, 75.0, 78.0, 74.0),
    ('ADT23SOCB0007', 25.0, 30.0, 40.0, 35.0, 38.0, 30.0),
    ('ADT23SOCB0008', 55.0, 60.0, 58.0, 62.0, 60.0, 55.0),
    ('ADT23SOCB0009', 100.0, 98.0, 99.0, 97.0, 100.0, 99.0),
    ('ADT23SOCB0010', 45.0, 50.0, 48.0, 55.0, 53.0, 50.0),
    ('ADT23SOCB0011', 80.0, 85.0, 82.0, 84.0, 83.0, 85.0),
])
conn.close()