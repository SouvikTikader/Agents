# [file name]: insert_100_students.py
# [file content begin]
import sqlite3
import random
from werkzeug.security import generate_password_hash
from datetime import datetime, timedelta

def insert_100_students():
    """Insert 100 student records with random performance data"""
    
    # Connect to database
    conn = sqlite3.connect('database/system.db')
    conn.row_factory = sqlite3.Row
    
    print("🎯 Starting to insert 100 student records...")
    
    # List of sample first names and last names for realistic usernames
    first_names = ['John', 'Jane', 'Alex', 'Emily', 'Michael', 'Sarah', 'David', 'Lisa', 
                  'Robert', 'Maria', 'William', 'Anna', 'James', 'Sofia', 'Daniel', 'Emma',
                  'Thomas', 'Olivia', 'Matthew', 'Ava', 'Christopher', 'Mia', 'Andrew', 'Isabella',
                  'Joshua', 'Amelia', 'Ryan', 'Charlotte', 'Jacob', 'Harper', 'Kevin', 'Evelyn',
                  'Brian', 'Abigail', 'Jonathan', 'Ella', 'Nathan', 'Scarlett', 'Justin', 'Luna',
                  'Brandon', 'Grace', 'Samuel', 'Chloe', 'Christian', 'Penelope', 'Benjamin', 'Riley',
                  'Zachary', 'Aria', 'Nicholas', 'Lily', 'Dylan', 'Nora', 'Tyler', 'Zoey', 'Jordan']
    
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
                 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
                 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
                 'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker',
                 'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores',
                 'Green', 'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 'Mitchell',
                 'Carter', 'Roberts']
    
    students_data = []
    performance_data = []
    
    # Generate 100 student records
    for i in range(1, 101):
        # Generate student ID in format ADT23SOCBXXXX
        student_id = f"ADT23SOCB{i:04d}"
        
        # Generate realistic email
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        email = f"{first_name.lower()}.{last_name.lower()}{i}@student.edu"
        
        # Default password (hashed)
        password_hash = generate_password_hash("student123")
        
        students_data.append((student_id, email, password_hash, 'student'))
        
        # Generate random performance data with realistic patterns
        
        # Attendance: Most students have good attendance, some have poor
        attendance_base = random.randint(60, 95)
        attendance_variation = random.randint(-10, 10)
        attendance_before_ta1 = max(40, min(100, attendance_base + attendance_variation))
        attendance_before_ta2 = max(40, min(100, attendance_base + random.randint(-5, 5)))
        attendance_before_final = max(40, min(100, attendance_base + random.randint(-5, 5)))
        
        # Marks: Create different performance patterns
        performance_type = random.choice(['excellent', 'good', 'average', 'poor', 'improving', 'declining'])
        
        if performance_type == 'excellent':
            marks_ta1 = random.randint(85, 98)
            marks_ta2 = random.randint(88, 99)
            assignment_marks = random.randint(90, 100)
            marks_final = random.randint(88, 100)
            
        elif performance_type == 'good':
            marks_ta1 = random.randint(75, 89)
            marks_ta2 = random.randint(78, 92)
            assignment_marks = random.randint(80, 95)
            marks_final = random.randint(75, 90)
            
        elif performance_type == 'average':
            marks_ta1 = random.randint(60, 79)
            marks_ta2 = random.randint(62, 82)
            assignment_marks = random.randint(65, 85)
            marks_final = random.randint(58, 78)
            
        elif performance_type == 'poor':
            marks_ta1 = random.randint(35, 59)
            marks_ta2 = random.randint(30, 62)
            assignment_marks = random.randint(40, 70)
            marks_final = random.randint(25, 55)
            
        elif performance_type == 'improving':
            marks_ta1 = random.randint(50, 65)
            marks_ta2 = random.randint(65, 80)
            assignment_marks = random.randint(70, 85)
            marks_final = random.randint(75, 90)
            
        else:  # declining
            marks_ta1 = random.randint(80, 90)
            marks_ta2 = random.randint(65, 79)
            assignment_marks = random.randint(60, 75)
            marks_final = random.randint(50, 70)
        
        # Add some random variation
        marks_ta1 += random.randint(-5, 5)
        marks_ta2 += random.randint(-5, 5)
        assignment_marks += random.randint(-5, 5)
        marks_final += random.randint(-5, 5)
        
        # Ensure marks are within bounds
        marks_ta1 = max(0, min(100, marks_ta1))
        marks_ta2 = max(0, min(100, marks_ta2))
        assignment_marks = max(0, min(100, assignment_marks))
        marks_final = max(0, min(100, marks_final))
        
        performance_data.append((
            student_id, 
            round(attendance_before_ta1, 1),
            round(marks_ta1, 1),
            round(attendance_before_ta2, 1),
            round(marks_ta2, 1),
            round(attendance_before_final, 1),
            round(assignment_marks, 1),
            round(marks_final, 1)
        ))
    
    try:
        # Insert students into users table
        print("📝 Inserting student accounts...")
        conn.executemany('''
            INSERT OR IGNORE INTO users (username, email, password, role)
            VALUES (?, ?, ?, ?)
        ''', students_data)
        
        # Insert performance data
        print("📊 Inserting performance records...")
        conn.executemany('''
            INSERT OR IGNORE INTO performance (
                student_id, attendance_before_ta1, marks_ta1,
                attendance_before_ta2, marks_ta2,
                attendance_before_final, assignment_marks, marks_final
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', performance_data)
        
        # Create some sample grievances for variety
        print("⚖️ Creating sample grievances...")
        grievance_categories = ['academic', 'administrative', 'faculty', 'infrastructure']
        grievance_messages = [
            "Need clarification on exam grading criteria",
            "Library resources are insufficient for my research",
            "Requesting extension for assignment deadline",
            "Issues with online learning platform access",
            "Concern about faculty feedback timeliness",
            "Classroom facilities need maintenance",
            "Request for additional tutoring sessions",
            "Issues with course registration system",
            "Need accommodation for learning disability",
            "Concern about course material quality"
        ]
        
        # Create grievances for some students
        sample_grievances = []
        for i in range(20):  # Create 20 sample grievances
            student_idx = random.randint(0, 99)
            student_id = students_data[student_idx][0]
            category = random.choice(grievance_categories)
            message = random.choice(grievance_messages)
            status = random.choice(['Pending', 'Resolved', 'In Progress'])
            priority = random.choice(['Low', 'Normal', 'High'])
            
            sample_grievances.append((
                student_id, category, message, status, priority
            ))
        
        conn.executemany('''
            INSERT INTO grievances (student_id, category, message, status, priority)
            VALUES (?, ?, ?, ?, ?)
        ''', sample_grievances)
        
        # Commit all changes
        conn.commit()
        
        # Print summary statistics
        print("\n📈 DATA INSERTION SUMMARY")
        print("=" * 40)
        
        # Count students by performance level
        excellent = len([p for p in performance_data if p[2] >= 85 and p[4] >= 85])
        good = len([p for p in performance_data if 70 <= p[2] < 85 and 70 <= p[4] < 85])
        average = len([p for p in performance_data if 50 <= p[2] < 70 and 50 <= p[4] < 70])
        poor = len([p for p in performance_data if p[2] < 50 or p[4] < 50])
        
        print(f"🎯 Total Students: {len(students_data)}")
        print(f"📊 Performance Distribution:")
        print(f"   Excellent (85+): {excellent} students")
        print(f"   Good (70-84): {good} students") 
        print(f"   Average (50-69): {average} students")
        print(f"   Poor (<50): {poor} students")
        
        # Calculate averages
        avg_ta1 = sum(p[2] for p in performance_data) / len(performance_data)
        avg_ta2 = sum(p[4] for p in performance_data) / len(performance_data)
        avg_final = sum(p[7] for p in performance_data) / len(performance_data)
        avg_attendance = sum(p[1] for p in performance_data) / len(performance_data)
        
        print(f"📐 Averages:")
        print(f"   TA1 Marks: {avg_ta1:.1f}%")
        print(f"   TA2 Marks: {avg_ta2:.1f}%")
        print(f"   Final Marks: {avg_final:.1f}%")
        print(f"   Attendance: {avg_attendance:.1f}%")
        
        print(f"⚖️ Sample Grievances: {len(sample_grievances)} created")
        
        print("\n✅ Successfully inserted 100 student records with realistic data!")
        print("🎮 The autonomous agents now have substantial data to analyze!")
        
    except Exception as e:
        print(f"❌ Error inserting data: {e}")
        conn.rollback()
    finally:
        conn.close()

def verify_data():
    """Verify the inserted data"""
    conn = sqlite3.connect('database/system.db')
    conn.row_factory = sqlite3.Row
    
    try:
        # Count students
        student_count = conn.execute("SELECT COUNT(*) as count FROM users WHERE role='student'").fetchone()['count']
        performance_count = conn.execute("SELECT COUNT(*) as count FROM performance").fetchone()['count']
        grievance_count = conn.execute("SELECT COUNT(*) as count FROM grievances").fetchone()['count']
        
        print(f"\n🔍 DATA VERIFICATION:")
        print(f"   Students in users table: {student_count}")
        print(f"   Performance records: {performance_count}")
        print(f"   Grievances: {grievance_count}")
        
        # Show some sample data
        print(f"\n👀 SAMPLE STUDENTS (first 5):")
        sample_students = conn.execute('''
            SELECT u.username, p.marks_ta1, p.marks_ta2, p.marks_final, p.attendance_before_ta1
            FROM users u 
            JOIN performance p ON u.username = p.student_id 
            WHERE u.role = 'student'
            LIMIT 5
        ''').fetchall()
        
        for student in sample_students:
            print(f"   {student['username']}: TA1={student['marks_ta1']}%, TA2={student['marks_ta2']}%, "
                  f"Final={student['marks_final']}%, Att={student['attendance_before_ta1']}%")
                  
    except Exception as e:
        print(f"❌ Verification error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    insert_100_students()
    verify_data()
# [file content end]