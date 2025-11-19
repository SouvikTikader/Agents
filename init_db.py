# [file name]: init_db.py
# [file content begin]
import sqlite3
import os

# Ensure database directory exists
os.makedirs("database", exist_ok=True)

# Connect to SQLite DB with foreign keys enabled
conn = sqlite3.connect("database/system.db")
conn.execute("PRAGMA foreign_keys = ON;")  # Enable FK constraints

# ---------------------------
# USERS TABLE
# ---------------------------
conn.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    password TEXT NOT NULL,
    role TEXT CHECK(role IN ('student', 'admin')) DEFAULT 'student'
)''')

# Create academic_analysis table
conn.execute("""
CREATE TABLE IF NOT EXISTS academic_analysis (
    student_id TEXT PRIMARY KEY,
    risk_level TEXT,
    risk_score REAL,
    recommendations TEXT
)
""")

# ---------------------------
# STUDENT PERFORMANCE TABLE
# ---------------------------
conn.execute('''CREATE TABLE IF NOT EXISTS performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    attendance_before_ta1 REAL CHECK(attendance_before_ta1 >= 0 AND attendance_before_ta1 <= 100) DEFAULT 0,
    marks_ta1 REAL CHECK(marks_ta1 >= 0 AND marks_ta1 <= 100) DEFAULT 0,
    attendance_before_ta2 REAL CHECK(attendance_before_ta2 >= 0 AND attendance_before_ta2 <= 100) DEFAULT 0,
    marks_ta2 REAL CHECK(marks_ta2 >= 0 AND marks_ta2 <= 100) DEFAULT 0,
    attendance_before_final REAL CHECK(attendance_before_final >= 0 AND attendance_before_final <= 100) DEFAULT 0,
    assignment_marks REAL CHECK(assignment_marks >= 0 AND assignment_marks <= 100) DEFAULT 0,
    marks_final REAL CHECK(marks_final >= 0 AND marks_final <= 100) DEFAULT 0,
    FOREIGN KEY (student_id) REFERENCES users(username) ON DELETE CASCADE ON UPDATE CASCADE
)''')

# Index on student_id for fast queries
conn.execute('CREATE INDEX IF NOT EXISTS idx_performance_student ON performance(student_id)')

# ---------------------------
# GRIEVANCES TABLE
# ---------------------------
conn.execute('''CREATE TABLE IF NOT EXISTS grievances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    category TEXT,
    message TEXT,
    status TEXT DEFAULT 'Pending',
    priority TEXT DEFAULT 'Normal',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES users(username) ON DELETE CASCADE ON UPDATE CASCADE
)''')

# Index on student_id for grievances
conn.execute('CREATE INDEX IF NOT EXISTS idx_grievances_student ON grievances(student_id)')

# ---------------------------
# NOTIFICATIONS TABLE (UPDATED)
# ---------------------------
conn.execute('''CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,  -- NULL means global notification
    message TEXT NOT NULL,
    notification_type TEXT DEFAULT 'system',  -- system, agent, alert, etc.
    priority TEXT DEFAULT 'normal',  -- low, normal, high, critical
    seen INTEGER DEFAULT 0,
    agent_source TEXT,  -- Which agent created this notification
    metadata TEXT,  -- JSON data for additional context
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE
)''')

# Index on user_id for notifications
conn.execute('CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id)')
conn.execute('CREATE INDEX IF NOT EXISTS idx_notifications_type ON notifications(notification_type)')

# ---------------------------
# AGENT MEMORY TABLE (ENHANCED)
# ---------------------------
conn.execute('''CREATE TABLE IF NOT EXISTS agent_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    memory_type TEXT CHECK(memory_type IN ('reflection', 'success_pattern', 'failure_pattern', 'intervention', 'student_profile')) NOT NULL,
    content TEXT NOT NULL,
    student_id TEXT,
    context_data TEXT,  -- JSON context when memory was stored
    confidence_score REAL DEFAULT 1.0,
    usage_count INTEGER DEFAULT 0,
    last_used DATETIME,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
)''')

# ---------------------------
# AGENT ACTIONS LOG (ENHANCED)
# ---------------------------
conn.execute('''CREATE TABLE IF NOT EXISTS agent_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    action_type TEXT NOT NULL,
    student_id TEXT,
    target_user TEXT,  -- Who this action affects
    details TEXT NOT NULL,
    context_before TEXT,  -- JSON state before action
    context_after TEXT,   -- JSON state after action
    success_score REAL DEFAULT 0.0,
    learned_lessons TEXT,
    status TEXT DEFAULT 'completed',
    result TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)''')

# ---------------------------
# INTERVENTION PLANS
# ---------------------------
conn.execute('''CREATE TABLE IF NOT EXISTS intervention_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    plan_type TEXT NOT NULL,
    actions TEXT NOT NULL,
    assigned_agent TEXT,
    status TEXT DEFAULT 'active',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME
)''')

conn.commit()
conn.close()

print("✅ Database initialized successfully at 'database/system.db' with enhanced schema.")
