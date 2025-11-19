# main.py 
from fastapi import FastAPI, Request, Form, Depends, HTTPException, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from agents.academic_agent import AcademicAgent
import sqlite3, pickle, pandas as pd
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
import threading, time
from datetime import datetime, timedelta
from textblob import TextBlob
import asyncio
from contextlib import asynccontextmanager
import uuid
import os
from agents.orchestrator import AgentOrchestrator
# Add to imports
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ADMIN_EMAILS = os.getenv('ADMIN_EMAILS', 'souviktikader077@gmail.com').split(',')
SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'souviktikader077@gmail.com')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', 'ypkw gomz xijp wkcx')

# Keep all your existing constants
DB_PATH = Path(__file__).parent / "database" / "system.db"
MODEL_PATH = Path(__file__).parent / "models" / "risk_model.pkl"

# Session storage 
sessions = {}
CATEGORY_KEYWORDS = {
    'academic': ['exam', 'marks', 'grade', 'grades', 'ta1', 'ta2', 'final', 'assignment', 'gpa', 'syllabus'],
    'administrative': ['hostel', 'fee', 'admission', 'registration', 'schedule', 'timetable', 'enrollment'],
    'faculty': ['professor', 'teacher', 'faculty', 'conduct', 'behavior', 'absent', 'attendance'],
    'infrastructure': ['lab', 'classroom', 'projector', 'wifi', 'facility']
}
agent_orchestrator = AgentOrchestrator()
async def run_academic_analysis_once():
    """Run AcademicAgent on all students and save analysis results"""
    agent = AcademicAgent()
    conn = get_db_connection()
    students = conn.execute("SELECT * FROM performance").fetchall()
    conn.close()

    print(f"🎓 Analyzing {len(students)} students for academic risk...")

    for s in students:
        student_data = dict(s)
        risk_result = await agent.calculate_student_risk_score(student_data)
        agent.save_risk_analysis(student_data.get("student_id", "UNKNOWN"), risk_result)


def academic_analysis_loop(interval=300):
    """Background loop that runs AcademicAgent periodically"""
    async def run_agent():
        while True:
            try:
                print(f"\n📘 [AcademicAgent Loop] Running academic risk analysis at {datetime.now().strftime('%H:%M:%S')}")
                await run_academic_analysis_once()
                print("✅ Academic risk analysis completed.\n")
            except Exception as e:
                print(f"⚠️ AcademicAgent loop error: {e}")
            await asyncio.sleep(interval)  # Run every 5 minutes

    asyncio.run(run_agent())
def risk_agent_loop(interval=300):
    """Analyze student risk and trigger alerts"""
    while True:
        try:
            thresholds = compute_dynamic_thresholds()
            with get_db_connection() as conn:
                students = conn.execute("SELECT * FROM performance").fetchall()

                for s in students:
                    student_id = s['student_id']
                    risk_reasons = []

                    # Threshold comparisons
                    ta1_t, ta2_t, final_t, assign_t, att_ta1_t, att_ta2_t, att_final_t = thresholds

                    if s['marks_ta1'] < ta1_t:
                        risk_reasons.append(f"Low TA1 marks ({s['marks_ta1']}%)")
                    if s['marks_ta2'] < ta2_t:
                        risk_reasons.append(f"Low TA2 marks ({s['marks_ta2']}%)")
                    if s['marks_final'] < final_t:
                        risk_reasons.append(f"Low Final marks ({s['marks_final']}%)")
                    if s['assignment_marks'] < assign_t:
                        risk_reasons.append(f"Low Assignment marks ({s['assignment_marks']}%)")
                    if s['attendance_before_ta1'] < att_ta1_t:
                        risk_reasons.append(f"Low Attendance before TA1 ({s['attendance_before_ta1']}%)")
                    if s['attendance_before_ta2'] < att_ta2_t:
                        risk_reasons.append(f"Low Attendance before TA2 ({s['attendance_before_ta2']}%)")
                    if s['attendance_before_final'] < att_final_t:
                        risk_reasons.append(f"Low Attendance before Final ({s['attendance_before_final']}%)")

                    if risk_reasons:
                        reasons_text = "; ".join(risk_reasons)
                        existing = conn.execute(
                            "SELECT * FROM notifications WHERE message LIKE ?",
                            (f"%{student_id}%at risk%",)
                        ).fetchone()

                        if not existing:
                            add_notification(
                                message=f"Academic Risk Alert for {student_id}: {reasons_text}",
                                user_id=student_id,
                                notification_type="risk_alert",
                                priority="high",
                                agent_source="RiskAgent"
                            )
                            add_notification(
                                message=f"High Risk Student: {student_id} - {reasons_text}",
                                user_id=None,
                                notification_type="risk_alert",
                                priority="high",
                                agent_source="RiskAgent"
                            )

            time.sleep(interval)
        except Exception as e:
            print(f"[RiskAgent Error] {e}")
            time.sleep(interval)

def grievance_agent_loop(interval=1200):
    """
    Monitor and auto-escalate repeated grievances + notify admins.
    Runs every 20 minutes by default.
    """
    while True:
        try:
            with get_db_connection() as conn:
                # Fetch all pending grievances
                pending = conn.execute(
                    "SELECT * FROM grievances WHERE status='Pending'"
                ).fetchall()

                for g in pending:
                    grievance_id = g['id']
                    student_id = g['student_id']
                    category = g['category']
                    message = g['message']

                    # 🔁 Check for duplicates of same message/category/student
                    duplicates = conn.execute(
                        """
                        SELECT COUNT(*) AS count FROM grievances
                        WHERE student_id=? AND category=? 
                        AND LOWER(TRIM(message))=LOWER(TRIM(?))
                        """,
                        (student_id, category, message)
                    ).fetchone()

                    # 🚨 If same grievance submitted 3 or more times — escalate & merge
                    if duplicates and duplicates['count'] >= 3:
                        escalate_grievance_priority(grievance_id, student_id, category)

                    # 🚨 Notify admins if high priority
                    if g['priority'].lower() == "high":
                        existing = conn.execute(
                            "SELECT * FROM notifications WHERE message LIKE ?",
                            (f"%Grievance #{g['id']}%High%",)
                        ).fetchone()
                        if not existing:
                            add_notification(
                                message=f"🚨 Urgent Grievance #{g['id']} from {g['student_id']} requires immediate attention",
                                user_id=None,
                                notification_type="grievance_urgent",
                                priority="high",
                                agent_source="GrievanceAgent"
                            )

            print(f"[GrievanceAgent] Completed scan at {datetime.now().strftime('%H:%M:%S')}")
            time.sleep(interval)

        except Exception as e:
            print(f"[GrievanceAgent Error] {e}")
            time.sleep(interval)


# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start your existing background threads
    threading.Thread(target=risk_agent_loop, daemon=True).start()
    threading.Thread(target=grievance_agent_loop, daemon=True).start()
    # Start Academic Agent background thread
    threading.Thread(target=academic_analysis_loop, daemon=True).start()

    
    loop = asyncio.get_event_loop()
    loop.create_task(agent_orchestrator.start_autonomous_operation())

    
    yield
    
    # Cleanup
    agent_orchestrator.stop_autonomous_operation()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== KEEP ALL YOUR EXISTING HELPER FUNCTIONS ==========

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


# [file name]: main.py
# Add these enhanced notification functions

def add_notification(message: str, user_id: str = None, notification_type: str = "system", 
                   priority: str = "normal", agent_source: str = None, metadata: dict = None):
    """Add notification with user targeting and agent tracking"""
    try:
        conn = get_db_connection()
        
        # Convert metadata to JSON string if provided
        metadata_json = None
        if metadata:
            import json
            metadata_json = json.dumps(metadata)
        
        conn.execute(
            """INSERT INTO notifications 
               (user_id, message, notification_type, priority, agent_source, metadata, timestamp) 
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
            (user_id, message, notification_type, priority, agent_source, metadata_json)
        )
        conn.commit()
        conn.close()
        
        print(f"📢 NOTIFICATION [{priority.upper()}] for {user_id or 'ALL'}: {message}")
        return True
        
    except Exception as e:
        print(f"❌ NOTIFICATION ERROR: {e}")
        return False

def log_agent_action(agent_name: str, action_type: str, details: str, student_id: str = None,
                    target_user: str = None, context_before: dict = None, context_after: dict = None,
                    success_score: float = 0.0, learned_lessons: str = None):
    """Log detailed agent actions for tracking and learning"""
    try:
        conn = get_db_connection()
        
        # Convert context to JSON
        import json
        context_before_json = json.dumps(context_before) if context_before else None
        context_after_json = json.dumps(context_after) if context_after else None
        
        conn.execute(
            """INSERT INTO agent_actions 
               (agent_name, action_type, student_id, target_user, details, 
                context_before, context_after, success_score, learned_lessons, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            (agent_name, action_type, student_id, target_user, details,
             context_before_json, context_after_json, success_score, learned_lessons)
        )
        conn.commit()
        conn.close()
        
        print(f"🤖 AGENT ACTION: {agent_name} -> {action_type} for {student_id or 'system'}")
        return True
        
    except Exception as e:
        print(f"❌ AGENT ACTION LOG ERROR: {e}")
        return False

def store_agent_memory(agent_name: str, memory_type: str, content: str, student_id: str = None,
                      context_data: dict = None, confidence_score: float = 1.0, metadata: dict = None):
    """Store agent memories for long-term learning"""
    try:
        conn = get_db_connection()
        
        import json
        context_json = json.dumps(context_data) if context_data else None
        metadata_json = json.dumps(metadata) if metadata else None
        
        conn.execute(
            """INSERT INTO agent_memory 
               (agent_name, memory_type, content, student_id, context_data, 
                confidence_score, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'), ?)""",
            (agent_name, memory_type, content, student_id, context_json, confidence_score, metadata_json)
        )
        conn.commit()
        conn.close()
        
        print(f"🧠 AGENT MEMORY: {agent_name} stored {memory_type} for {student_id or 'general'}")
        return True
        
    except Exception as e:
        print(f"❌ AGENT MEMORY STORAGE ERROR: {e}")
        return False

def get_agent_memories(agent_name: str, memory_type: str = None, student_id: str = None, limit: int = 10):
    """Retrieve agent memories for decision making"""
    try:
        conn = get_db_connection()
        
        query = "SELECT * FROM agent_memory WHERE agent_name = ?"
        params = [agent_name]
        
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)
            
        if student_id:
            query += " AND student_id = ?"
            params.append(student_id)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        memories = conn.execute(query, params).fetchall()
        conn.close()
        
        return [dict(memory) for memory in memories]
        
    except Exception as e:
        print(f"❌ AGENT MEMORY RETRIEVAL ERROR: {e}")
        return []

def get_suggestions_from_memory(agent_name: str, student_id: str = None):
    """Get suggestions based on past agent memories and patterns"""
    try:
        memories = get_agent_memories(agent_name, student_id=student_id, limit=20)
        
        if not memories:
            return ["No historical data available for suggestions."]
        
        # Analyze memory patterns to generate suggestions
        success_patterns = [m for m in memories if m['memory_type'] == 'success_pattern']
        failure_patterns = [m for m in memories if m['memory_type'] == 'failure_pattern']
        interventions = [m for m in memories if m['memory_type'] == 'intervention']
        
        suggestions = []
        
        # Generate suggestions based on patterns
        if success_patterns:
            recent_success = success_patterns[0]
            suggestions.append(f"✅ Repeat successful strategy: {recent_success['content'][:100]}...")
        
        if failure_patterns:
            recent_failure = failure_patterns[0]
            suggestions.append(f"⚠️ Avoid previous failure: {recent_failure['content'][:100]}...")
        
        if interventions:
            successful_interventions = [i for i in interventions if i.get('confidence_score', 0) > 0.7]
            if successful_interventions:
                suggestions.append(f"🎯 Consider proven intervention: {successful_interventions[0]['content'][:100]}...")
        
        # Add general suggestions based on memory count
        if len(memories) > 10:
            suggestions.append("📊 You have substantial historical data - trust your learned patterns")
        
        if not suggestions:
            suggestions.append("🔍 Continue monitoring and gathering data for better suggestions")
        
        return suggestions
        
    except Exception as e:
        print(f"❌ SUGGESTION GENERATION ERROR: {e}")
        return ["Error generating suggestions from memory."]
    
    
def extract_email_subject(self, message: str) -> str:
    """Extract a concise subject from notification message"""
    if '🚨' in message:
        return "CRITICAL ALERT - Immediate Action Required"
    elif 'COLLABORATION' in message:
        return "Agent Collaboration Alert"
    elif 'CRITICAL' in message:
        return "Critical Risk Detected"
    elif 'PREDICTIVE' in message:
        return "Predictive Analytics Alert"
    else:
        return "System Notification"

# Add new API endpoints for testing email
@app.get("/test-email")
async def test_email():
    """Test email notification system"""
    test_message = "🤖 TEST: This is a test of the autonomous agent email notification system."
    
    success = add_notification(
        test_message, 
        send_email=True,
        email_priority="medium"
    )
    
    return {
        "success": success,
        "message": test_message,
        "email_sent": success,
        "email_system_enabled": email_notifier.enabled
    }

@app.get("/test-critical-email")
async def test_critical_email():
    """Test critical email notification"""
    test_message = "🚨 CRITICAL TEST: This is a test of critical email alerts. Immediate action may be required."
    
    success = add_notification(
        test_message,
        send_email=True, 
        email_priority="high"
    )
    
    return {
        "success": success,
        "message": test_message,
        "email_sent": success,
        "priority": "high"
    }

def compute_dynamic_thresholds():
    conn = get_db_connection()
    def avg_or_default(query, default):
        val = conn.execute(query).fetchone()[0]
        return (val if val is not None else default) * 0.8

    ta1_marks_threshold = avg_or_default("SELECT AVG(marks_ta1) FROM performance", 50)
    ta2_marks_threshold = avg_or_default("SELECT AVG(marks_ta2) FROM performance", 50)
    final_marks_threshold = avg_or_default("SELECT AVG(marks_final) FROM performance", 50)
    assignment_marks_threshold = avg_or_default("SELECT AVG(assignment_marks) FROM performance", 50)

    att_ta1_threshold = avg_or_default("SELECT AVG(attendance_before_ta1) FROM performance", 60)
    att_ta2_threshold = avg_or_default("SELECT AVG(attendance_before_ta2) FROM performance", 60)
    att_final_threshold = avg_or_default("SELECT AVG(attendance_before_final) FROM performance", 60)
    conn.close()

    return (
        ta1_marks_threshold, ta2_marks_threshold, final_marks_threshold, assignment_marks_threshold,
        att_ta1_threshold, att_ta2_threshold, att_final_threshold
    )

def analyze_sentiment(text):
    category = "general"
    for cat, keys in CATEGORY_KEYWORDS.items():
        for k in keys:
            if k.lower() in text.lower():
                category = cat
                break

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity < -0.3 or subjectivity > 0.6:
        urgency = "High"
    elif polarity < 0:
        urgency = "Medium"
    else:
        urgency = "Low"

    return category, urgency

def escalate_grievance_priority(grievance_id, student_id, category):
    """
    Merge identical grievances (same student, category, and message ≥3 times),
    set merged one to HIGH priority,
    and auto-close duplicates when main grievance is resolved.
    """
    with get_db_connection() as conn:
        # Get main grievance message
        msg_row = conn.execute("SELECT message, status FROM grievances WHERE id=?", (grievance_id,)).fetchone()
        if not msg_row:
            return

        message_text = msg_row["message"].strip().lower()
        current_status = msg_row["status"]

        # Find duplicates of same message/category/student
        duplicates = conn.execute(
            """
            SELECT id FROM grievances
            WHERE student_id=? AND category=? 
            AND LOWER(TRIM(message))=? AND id != ?
            """,
            (student_id, category, message_text, grievance_id),
        ).fetchall()

        duplicate_ids = [d["id"] for d in duplicates]

        # --- Case 1: Merge if 3 or more identical grievances ---
        if len(duplicate_ids) + 1 >= 3:
            # Keep the newest one
            all_ids = duplicate_ids + [grievance_id]
            main_id = max(all_ids)
            old_ids = [str(i) for i in all_ids if i != main_id]

            # Delete duplicates from table
            if old_ids:
                placeholders = ",".join(["?"] * len(old_ids))
                conn.execute(f"DELETE FROM grievances WHERE id IN ({placeholders})", old_ids)

            # Escalate merged grievance
            conn.execute(
                "UPDATE grievances SET priority='High' WHERE id=?",
                (main_id,),
            )
            conn.commit()

            print(f"⚠️ Auto-merge: Student {student_id} had {len(all_ids)} identical grievances → merged into #{main_id} (HIGH priority)")

            add_notification(
                message=f"Grievances merged: Student {student_id} submitted same issue {len(all_ids)} times → merged into one HIGH priority grievance (#{main_id})",
                user_id=None,
                notification_type="grievance_merge_escalation",
                priority="high",
                agent_source="System"
            )

        # --- Case 2: Auto-resolve duplicates when main grievance is resolved ---
        if current_status.lower() == "resolved":
            resolved_duplicates = conn.execute(
                """
                SELECT id FROM grievances
                WHERE student_id=? AND category=? 
                AND LOWER(TRIM(message))=? AND status='Pending'
                """,
                (student_id, category, message_text),
            ).fetchall()
            if resolved_duplicates:
                ids_to_close = [str(r["id"]) for r in resolved_duplicates]
                placeholders = ",".join(["?"] * len(ids_to_close))
                conn.execute(f"UPDATE grievances SET status='Resolved' WHERE id IN ({placeholders})", ids_to_close)
                conn.commit()

                print(f"✅ Auto-resolved {len(ids_to_close)} duplicate grievances for {student_id} ({category})")

                add_notification(
                    message=f"✅ Auto-resolved {len(ids_to_close)} duplicate grievances for {student_id} after main grievance resolution.",
                    user_id=None,
                    notification_type="grievance_auto_resolve",
                    priority="normal",
                    agent_source="System"
                )

# ========== SESSION MANAGEMENT ==========

def get_session(request: Request):
    session_id = request.cookies.get("session_id")
    return sessions.get(session_id)

def require_login(request: Request):
    session_data = get_session(request)
    if not session_data:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return session_data

def require_admin(request: Request):
    session_data = require_login(request)
    if session_data.get('role') != 'admin':
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return session_data

# ========== AUTHENTICATION ROUTES ==========

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    session_data = get_session(request)
    if not session_data:
        return RedirectResponse(url="/login")
    
    # Redirect based on role
    if session_data.get('role') == 'admin':
        return RedirectResponse(url="/admin")
    else:
        return RedirectResponse(url="/dashboard")
    
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, message: str = None, error: str = None):
    return templates.TemplateResponse("login.html", {
        "request": request,
        "message": message,
        "error": error
    })

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    
    if user and check_password_hash(user['password'], password):
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            'user': username,
            'role': user['role']
        }
        
        # Redirect based on role
        if user['role'] == 'admin':
            redirect_url = "/admin"
        else:
            redirect_url = "/dashboard"
        
        response = RedirectResponse(url=redirect_url, status_code=303)
        response.set_cookie(key="session_id", value=session_id, httponly=True)
        return response
    else:
        return RedirectResponse(url="/login?error=Invalid+credentials", status_code=303)
@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, message: str = None, error: str = None):
    return templates.TemplateResponse("register.html", {
        "request": request,
        "message": message,
        "error": error
    })

@app.post("/register")
async def register(
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form("student")
):
    password_hash = generate_password_hash(password)
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
            (username, password_hash, role)
        )
        conn.commit()
        conn.close()
        return RedirectResponse(url="/login?message=Registration+successful", status_code=303)
    except Exception as e:
        conn.close()
        return RedirectResponse(url="/register?error=Username+already+exists", status_code=303)

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session_id")
    return response

# ========== STUDENT DASHBOARD ROUTES ==========

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    session_data = require_login(request)
    username = session_data['user']

    conn = get_db_connection()
    record = conn.execute(
        "SELECT * FROM performance WHERE student_id = ?", (username,)
    ).fetchone()
    conn.close()

    avg_marks, final_attendance = 0, 0

    if record:
        marks = [
            record['marks_ta1'] or 0,
            record['marks_ta2'] or 0,
            record['marks_final'] or 0,
            record['assignment_marks'] or 0
        ]
        valid_marks = [m for m in marks if m > 0]
        if valid_marks:
            avg_marks = round(sum(valid_marks) / len(valid_marks), 2)

        att_final = record['attendance_before_final'] or 0
        att_ta2 = record['attendance_before_ta2'] or 0
        att_ta1 = record['attendance_before_ta1'] or 0

        if att_final > 0:
            final_attendance = att_final
        elif att_ta2 > 0:
            final_attendance = att_ta2
        elif att_ta1 > 0:
            final_attendance = att_ta1
        else:
            final_attendance = 0

    return templates.TemplateResponse(
        'dashboard.html',
        {
            'request': request,
            'username': username,
            'record': dict(record) if record else None,
            'avg_marks': avg_marks,
            'avg_attendance': final_attendance
        }
    )



@app.get("/griev", response_class=HTMLResponse)
async def grievance_page(request: Request):
    session_data = require_login(request)
    student_id = session_data['user']
    username = session_data['user']

    conn = get_db_connection()
    past_grievances = conn.execute(
        "SELECT * FROM grievances WHERE student_id=? ORDER BY timestamp DESC",
        (student_id,)
    ).fetchall()
    conn.close()

    return templates.TemplateResponse(
        'submit_grievance.html', 
        {
            'request': request,
            'username': username,
            'past_grievances': [dict(g) for g in past_grievances]
        }
    )

@app.post("/submit_grievance")
async def submit_grievance(
    request: Request,
    message: str = Form(...)
):
    session_data = require_login(request)
    student_id = session_data['user']
    
    Category, urgency = analyze_sentiment(message)
    
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO grievances (student_id, category, message, status, priority) VALUES (?, ?, ?, ?, ?)",
        (student_id, Category, message, "Pending", urgency)
    )
    conn.commit()
    conn.close()

    # UPDATED: Add targeted notification
    add_notification(
        message=f"New grievance submitted by {student_id} ({Category}) - Priority: {urgency}",
        user_id=student_id,  # Notify the student
        notification_type="grievance",
        priority=urgency.lower(),
        agent_source="System"
    )
    
    # Also notify admins about new grievance
    add_notification(
        message=f"New grievance from {student_id}: {message[:50]}...",
        user_id=None,  # Global notification for admins
        notification_type="grievance_alert", 
        priority=urgency.lower(),
        agent_source="System"
    )
    return RedirectResponse(url="/griev?message=Grievance+submitted+successfully", status_code=303)

# ========== RISK PREDICTION ROUTES ==========


async def predict_risk_page(request: Request):
    session_data = require_login(request)
    username = session_data['user']  # Get username
    
    return templates.TemplateResponse('predict_risk.html', {
        'request': request,
        'username': username  # Add this line
    })
# Add this missing route in main.py
@app.get("/predict_risk_page", response_class=HTMLResponse)
async def predict_risk_page(request: Request):
    session_data = require_login(request)
    username = session_data['user']
    
    return templates.TemplateResponse('predict_risk.html', {
        'request': request,
        'username': username
    })

@app.post("/predict_risk")
async def predict_risk(
    request: Request,
    ta1_marks: float = Form(...),
    ta2_marks: float = Form(...),
    final_marks: float = Form(...),
    assignment_marks: float = Form(...),
    attendance: float = Form(...)
):
    session_data = require_login(request)
    student_username = session_data['user']

    thresholds = compute_dynamic_thresholds()

    components = [
        ("TA1", ta1_marks, thresholds[0]),
        ("TA2", ta2_marks, thresholds[1]),
        ("Final Exam", final_marks, thresholds[2]),
        ("Assignments", assignment_marks, thresholds[3]),
        ("Attendance", attendance, min(thresholds[4], thresholds[5], thresholds[6]))
    ]

    risk_summary = []
    for name, actual, thresh in components:
        if actual < thresh:
            status = "At Risk"
            suggestion = ""
            if name in ["TA1", "TA2", "Final Exam", "Assignments"]:
                suggestion = f"Focus on improving {name}, need ≥ {thresh:.1f} marks"
            else:  # Attendance
                suggestion = f"Increase attendance to at least {thresh:.1f}%"
        else:
            status = "Safe"
            suggestion = "Keep up the good work!"
        risk_summary.append({
            "component": name,
            "actual": actual,
            "threshold": thresh,
            "status": status,
            "suggestion": suggestion
        })

    return templates.TemplateResponse(
        'predict_risk.html', 
        {
            'request': request,
            'username': student_username,
            'risk_summary': risk_summary
        }
    )

# ========== ADMIN ROUTES ==========

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    require_admin(request)
    
    conn = get_db_connection()

    # Student & grievance data
    students = conn.execute("SELECT * FROM performance").fetchall()
    grievances = conn.execute("SELECT * FROM grievances").fetchall()

    # Counts
    total_students = len(students)
    pending_grievances = conn.execute("SELECT COUNT(*) FROM grievances WHERE status='Pending'").fetchone()[0]
    resolved_grievances = conn.execute("SELECT COUNT(*) FROM grievances WHERE status='Resolved'").fetchone()[0]

    # Count from academic_analysis table (used by risk dashboard)
    at_risk_count = conn.execute("SELECT COUNT(*) FROM academic_analysis WHERE risk_level IN ('HIGH', 'CRITICAL')").fetchone()[0]

    conn.close()

    # Pass all counts to template
    return templates.TemplateResponse(
        'admin.html',
        {
            'request': request,
            'students': [dict(s) for s in students],
            'grievances': [dict(g) for g in grievances],
            'total_students': total_students,
            'pending_grievances': pending_grievances,
            'resolved_grievances': resolved_grievances,
            'at_risk_count': at_risk_count
        }
    )

@app.get("/grievances", response_class=HTMLResponse)
async def grievances_page(request: Request):
    require_admin(request)

    conn = get_db_connection()
    grievances = conn.execute("SELECT * FROM grievances ORDER BY timestamp DESC").fetchall()

    pending_count = conn.execute("SELECT COUNT(*) FROM grievances WHERE status='Pending'").fetchone()[0]
    resolved_count = conn.execute("SELECT COUNT(*) FROM grievances WHERE status='Resolved'").fetchone()[0]
    rejected_count = conn.execute("SELECT COUNT(*) FROM grievances WHERE status='Rejected'").fetchone()[0]
    total_count = pending_count + resolved_count + rejected_count

    top_category = conn.execute("""
        SELECT category, COUNT(*) FROM grievances GROUP BY category ORDER BY COUNT(*) DESC LIMIT 1
    """).fetchone()
    top_category = top_category[0] if top_category else None

    top_student = conn.execute("""
        SELECT student_id, COUNT(*) FROM grievances GROUP BY student_id ORDER BY COUNT(*) DESC LIMIT 1
    """).fetchone()
    top_student = top_student[0] if top_student else None

    conn.close()

    return templates.TemplateResponse(
        'grievance.html',
        {
            'request': request,
            'grievances': [dict(g) for g in grievances],
            'pending_count': pending_count,
            'resolved_count': resolved_count,
            'rejected_count': rejected_count,
            'total_count': total_count,
            'top_category': top_category,
            'top_student': top_student,
        }
    )

@app.get("/update_status/{id}/{status}")
async def update_status(id: int, status: str, request: Request):
    require_admin(request)
    
    conn = get_db_connection()
    grievance = conn.execute("SELECT * FROM grievances WHERE id=?", (id,)).fetchone()
    conn.execute("UPDATE grievances SET status=? WHERE id=?", (status, id))
    conn.commit()
    conn.close()

    add_notification(
        message=f"✅ Grievance #{id} marked as {status}",
        user_id=None,  # Global for all admins
        notification_type="grievance_update",
        priority="normal",
        agent_source="System"
    )
    
    # Notify the student about their grievance status
    if grievance:
        add_notification(
            message=f"Your grievance #{id} has been {status.lower()}",
            user_id=grievance['student_id'],
            notification_type="grievance_update", 
            priority="normal",
            agent_source="System"
        )
    return RedirectResponse(url="/grievances", status_code=303)

@app.get("/add_marks", response_class=HTMLResponse)
async def add_marks_page(request: Request):
    require_admin(request)

    conn = get_db_connection()
    students = conn.execute("SELECT username FROM users WHERE role='student'").fetchall()
    conn.close()

    return templates.TemplateResponse('add_marks.html', {
        'request': request,
        'students': [dict(s) for s in students]
    })

@app.post("/add_marks")
async def add_marks(
    request: Request,
    action: str = Form(...),
    student_id: str = Form(None),
    new_student_id: str = Form(None),
    attendance_before_ta1: float = Form(None),
    marks_ta1: float = Form(None),
    attendance_before_ta2: float = Form(None),
    marks_ta2: float = Form(None),
    attendance_before_final: float = Form(None),
    assignment_marks: float = Form(None),
    marks_final: float = Form(None)
):
    require_admin(request)

    conn = get_db_connection()
    students = conn.execute("SELECT username FROM users WHERE role='student'").fetchall()

    if action == 'delete':
        conn.execute("DELETE FROM performance WHERE student_id=?", (student_id,))
        conn.execute("DELETE FROM users WHERE username=?", (student_id,))
        conn.commit()
        conn.close()
        return RedirectResponse(url="/admin", status_code=303)

    target_id = new_student_id if new_student_id else student_id

    if action in ('add', 'update'):
        if new_student_id:
            default_password = generate_password_hash("default123")
            try:
                conn.execute(
                    "INSERT INTO users (username, password, role) VALUES (?, ?, 'student')",
                    (new_student_id, default_password)
                )
            except sqlite3.IntegrityError:
                conn.close()
                return templates.TemplateResponse('add_marks.html', {
                    'request': request,
                    'students': [dict(s) for s in students],
                    'error': "Student ID already exists!"
                })

        existing = conn.execute("SELECT * FROM performance WHERE student_id=?", (target_id,)).fetchone()

        def parse_float_or_none(val):
            try:
                return float(val) if val is not None else None
            except (ValueError, TypeError):
                return None

        fields = {
            'attendance_before_ta1': parse_float_or_none(attendance_before_ta1),
            'marks_ta1': parse_float_or_none(marks_ta1),
            'attendance_before_ta2': parse_float_or_none(attendance_before_ta2),
            'marks_ta2': parse_float_or_none(marks_ta2),
            'attendance_before_final': parse_float_or_none(attendance_before_final),
            'assignment_marks': parse_float_or_none(assignment_marks),
            'marks_final': parse_float_or_none(marks_final)
        }

        for k in fields:
            if fields[k] is None:
                fields[k] = existing[k] if existing else 0

        if existing:
            conn.execute(
                '''UPDATE performance SET
                    attendance_before_ta1=?, marks_ta1=?, attendance_before_ta2=?, marks_ta2=?, 
                    attendance_before_final=?, assignment_marks=?, marks_final=?
                   WHERE student_id=?''',
                (*fields.values(), target_id)
            )
        else:
            conn.execute(
                '''INSERT INTO performance 
                (student_id, attendance_before_ta1, marks_ta1, attendance_before_ta2, marks_ta2,
                 attendance_before_final, assignment_marks, marks_final)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (target_id, *fields.values())
            )
        conn.commit()
        conn.close()
        return RedirectResponse(url="/admin", status_code=303)

    conn.close()
    return templates.TemplateResponse('add_marks.html', {
        'request': request,
        'students': [dict(s) for s in students]
    })

# ========== PERFORMANCE & NOTIFICATION ROUTES ==========

@app.get("/perf", response_class=HTMLResponse)
async def performance_dashboard(request: Request):
    require_login(request)
    
    conn = get_db_connection()
    students = conn.execute('''
        SELECT student_id, attendance_before_ta1, marks_ta1,
               attendance_before_ta2, marks_ta2,
               attendance_before_final, marks_final,
               assignment_marks
        FROM performance
    ''').fetchall()
    conn.close()

    students = [dict(row) for row in students]

    ta1_thresh, ta2_thresh, final_thresh, assignment_thresh, att_ta1_thresh, att_ta2_thresh, att_final_thresh = compute_dynamic_thresholds()

    def generate_ai_feedback(s):
        insights = []
        recommendations = []

        att_ta1 = s['attendance_before_ta1']
        att_ta2 = s['attendance_before_ta2']
        att_final = s['attendance_before_final']
        marks_ta1 = s['marks_ta1']
        marks_ta2 = s['marks_ta2']
        marks_final = s['marks_final']
        assign = s['assignment_marks']

        if att_ta1 == 0 or att_ta2 == 0:
            insights.append("Zero attendance detected - student may be disengaged or facing major issues.")
            recommendations.append("Immediately contact student and class representative for intervention.")
        elif att_ta1 < 40 or att_ta2 < 40:
            insights.append("Low attendance observed - risk of academic disengagement.")
            recommendations.append("Encourage attendance monitoring and personalized counselling.")

        avg_marks = (marks_ta1 + marks_ta2 + assign) / 3
        if avg_marks < 40:
            insights.append("Consistently poor academic performance - high risk of failure.")
            recommendations.append("Schedule remedial classes and assign a mentor.")
        elif 40 <= avg_marks < 60:
            insights.append("Moderate performance - needs consistent effort.")
            recommendations.append("Recommend participation in study groups and progress review.")
        elif avg_marks >= 80:
            insights.append("Excellent performance - strong academic standing.")
            recommendations.append("Nominate for advanced learning or peer tutoring roles.")

        if (marks_ta2 - marks_ta1) < -15:
            insights.append("Performance declined from TA1 to TA2 - possible learning gap.")
            recommendations.append("Review course topics between TA1 and TA2 and provide revision support.")
        elif (marks_ta2 - marks_ta1) > 15:
            insights.append("Performance improved from TA1 to TA2 - positive learning trend.")
            recommendations.append("Acknowledge progress and continue current learning strategies.")

        if assign < 40:
            insights.append("Low assignment engagement - weak concept application or time management.")
            recommendations.append("Encourage completion of additional practice assignments.")

        return insights, recommendations

    for s in students:
        s['risk'] = []
        if s['marks_ta1'] < ta1_thresh:
            s['risk'].append("TA1 Marks Low")
        if s['marks_ta2'] < ta2_thresh:
            s['risk'].append("TA2 Marks Low")
        if s['marks_final'] < final_thresh:
            s['risk'].append("Final Marks Low")
        if s['assignment_marks'] < assignment_thresh:
            s['risk'].append("Assignment Low")
        if s['attendance_before_ta1'] < att_ta1_thresh:
            s['risk'].append("TA1 Attendance Low")
        if s['attendance_before_ta2'] < att_ta2_thresh:
            s['risk'].append("TA2 Attendance Low")
        if s['attendance_before_final'] < att_final_thresh:
            s['risk'].append("Final Attendance Low")

        s['risk'] = "; ".join(s['risk']) if s['risk'] else "None"
        s['insights'], s['recommendations'] = generate_ai_feedback(s)

    return templates.TemplateResponse('perf.html', {
        'request': request,
        'students': students
    })

@app.get("/at_risk", response_class=HTMLResponse)
async def at_risk(request: Request):
    require_login(request)
    
    ta1_thresh, ta2_thresh, final_thresh, assignment_thresh, att_ta1_thresh, att_ta2_thresh, att_final_thresh = compute_dynamic_thresholds()

    conn = get_db_connection()
    students = conn.execute("SELECT * FROM performance").fetchall()
    conn.close()

    at_risk_students = []
    for s in students:
        if (s['marks_ta1'] < ta1_thresh or
            s['marks_ta2'] < ta2_thresh or
            s['marks_final'] < final_thresh or
            s['assignment_marks'] < assignment_thresh or
            s['attendance_before_ta1'] < att_ta1_thresh or
            s['attendance_before_ta2'] < att_ta2_thresh or
            s['attendance_before_final'] < att_final_thresh):
            at_risk_students.append(dict(s))
            
    return templates.TemplateResponse('at_risk.html', {
        'request': request,
        'students': at_risk_students
    })

# [file name]: main.py
# Update the notifications route

@app.get("/notifications", response_class=HTMLResponse)
async def notifications_page(request: Request):
    session_data = require_login(request)
    username = session_data["user"]
    user_role = session_data["role"]

    conn = get_db_connection()

    if user_role == "admin":
        # Admin sees all notifications
        notes = conn.execute(
            "SELECT * FROM notifications ORDER BY timestamp DESC"
        ).fetchall()
        # Mark all as read for admin
        conn.execute("UPDATE notifications SET seen = 1 WHERE seen = 0")
    else:
        # Student sees only their personal notifications
        notes = conn.execute(
            "SELECT * FROM notifications WHERE user_id = ? ORDER BY timestamp DESC",
            (username,)
        ).fetchall()
        # ✅ Mark student's notifications as read
        conn.execute(
            "UPDATE notifications SET seen = 1 WHERE user_id = ? AND seen = 0",
            (username,)
        )

    conn.commit()
    conn.close()

    return templates.TemplateResponse(
        "notifications.html",
        {
            "request": request,
            "notes": notes,
            "username": username,
            "user_role": user_role,
        },
    )

# Add agent monitoring endpoints
@app.get("/api/agent-system/status")
async def get_agent_system_status():
    return await agent_orchestrator.get_agent_status()

@app.post("/api/agent-system/restart")
async def restart_agent_system():
    agent_orchestrator.stop_autonomous_operation()
    await asyncio.sleep(2)
    asyncio.create_task(agent_orchestrator.start_autonomous_operation())
    return {"message": "Agent system restarting..."}

@app.get("/api/agent-system/performance")
async def get_agent_performance():
    status = await agent_orchestrator.get_agent_status()
    performance_data = {}
    
    for agent_name, agent_data in status["agents"].items():
        performance_data[agent_name] = {
            "goals_achieved": agent_data["performance_metrics"]["goals_achieved"],
            "plans_executed": agent_data["performance_metrics"]["plans_executed"],
            "success_rate": agent_data["performance_metrics"]["success_rate"],
            "learning_cycles": agent_data["memory_usage"]["learning_cycles"]
        }
    
    return performance_data
@app.get("/agent-monitor", response_class=HTMLResponse)
async def agent_monitor(request: Request):
    """Real-time agent monitoring dashboard"""
    require_admin(request)
    return templates.TemplateResponse('agent_monitor.html', {"request": request})

@app.get("/api/enhanced-context")
async def get_enhanced_context():
    """Get enhanced context with predictive analytics"""
    context = await agent_orchestrator.gather_system_context()
    
    # Get recent activities from agent memories
    recent_activities = []
    for agent_name, agent in agent_orchestrator.agents.items():
        recent_reflections = agent.memory.intervention_history[-5:]  # Last 5
        for reflection in recent_reflections:
            recent_activities.append({
                "agent": agent_name,
                "action": reflection.get("plan", {}).get("goal", "unknown"),
                "timestamp": reflection.get("timestamp"),
                "success": reflection.get("success", False)
            })
    
    return {
        "academic_context": context.get('academic_context', {}),
        "recent_activities": sorted(recent_activities, key=lambda x: x['timestamp'], reverse=True)[:10]
    }

@app.get("/debug-risk-analysis")
async def debug_risk_analysis():
    """Debug risk analysis for all students"""
    context = await agent_orchestrator.gather_system_context()
    academic_context = context.get('academic_context', {})
    
    return {
        "student_risk_analysis": academic_context.get('student_risk_analysis', []),
        "predictive_metrics": academic_context.get('predictive_metrics', {}),
        "risk_counts": {
            "critical": academic_context.get('critical_risk_count', 0),
            "high": academic_context.get('high_risk_count', 0),
            "medium": academic_context.get('medium_risk_count', 0),
            "total": len(academic_context.get('student_risk_analysis', []))
        }
    }

@app.get("/api/students")
async def api_students(request: Request):
    session_data = require_login(request)
    if session_data.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    conn = get_db_connection()
    students = conn.execute("SELECT * FROM performance").fetchall()
    conn.close()
    
    return {"students": [dict(s) for s in students]}

@app.get("/api/grievances")
async def api_grievances(request: Request):
    session_data = require_login(request)
    if session_data.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    conn = get_db_connection()
    grievances = conn.execute("SELECT * FROM grievances ORDER BY timestamp DESC").fetchall()
    conn.close()
    
    return {"grievances": [dict(g) for g in grievances]}


# Add these endpoints
@app.get("/agent-dashboard", response_class=HTMLResponse)
async def agent_dashboard(request: Request):
    require_admin(request)
    return templates.TemplateResponse('agent_dashboard.html', {"request": request})

@app.get("/api/student-interventions")
async def get_student_interventions(request: Request):
    session_data = require_login(request)
    student_id = session_data['user']
    
    # Get interventions for this student
    interventions = await agent_orchestrator.get_student_interventions(student_id)
    return JSONResponse(interventions)

@app.get("/api/agent-interventions")
async def get_agent_interventions(request: Request):
    require_admin(request)
    # Return recent interventions from agents
    interventions = await agent_orchestrator.get_recent_interventions()
    return JSONResponse(interventions)

@app.post("/api/ai-predictions")
async def get_ai_predictions(request: Request):
    session_data = require_login(request)
    data = await request.json()
    
    # Use academic agent to generate predictions
    predictions = await agent_orchestrator.agents['academic'].generate_predictions(data)

# [file name]: main.py
# Add these endpoints to main.py

@app.get("/api/agent/memories/{agent_name}")
async def get_agent_memories_api(agent_name: str, request: Request):
    """Get memories for a specific agent"""
    require_admin(request)
    
    memories = get_agent_memories(agent_name, limit=20)
    return JSONResponse({"agent": agent_name, "memories": memories})

@app.get("/api/agent/actions/{agent_name}")
async def get_agent_actions_api(agent_name: str, request: Request):
    """Get action log for a specific agent"""
    require_admin(request)
    
    conn = get_db_connection()
    actions = conn.execute(
        "SELECT * FROM agent_actions WHERE agent_name = ? ORDER BY timestamp DESC LIMIT 50",
        (agent_name,)
    ).fetchall()
    conn.close()
    
    return JSONResponse({"agent": agent_name, "actions": [dict(a) for a in actions]})

@app.get("/api/agent/suggestions/{agent_name}")
async def get_agent_suggestions_api(agent_name: str, student_id: str = None, request: Request = None):
    """Get AI suggestions based on agent memory"""
    if request:
        require_admin(request)
    
    suggestions = get_suggestions_from_memory(agent_name, student_id)
    return JSONResponse({"agent": agent_name, "student_id": student_id, "suggestions": suggestions})
    return JSONResponse(predictions)

@app.post("/send_message")
async def send_message(request: Request):
    data = await request.json()
    student_id = data.get("student_id")
    message = data.get("message")

    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO notifications (user_id, message, notification_type, priority, agent_source, timestamp)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
        """,
        (student_id, message, "direct_message", "normal", "AdminPanel")
    )
    conn.commit()
    conn.close()

    print(f"📨 Personal message sent to {student_id}: {message}")
    return {"status": "success"}

@app.get("/unread_count")
async def unread_count(request: Request):
    session_data = require_login(request)
    username = session_data["user"]
    role = session_data["role"]

    conn = get_db_connection()
    if role == "admin":
        # Admin sees all unread
        count = conn.execute("SELECT COUNT(*) FROM notifications WHERE seen = 0").fetchone()[0]
    else:
        # Students see only personal unread
        count = conn.execute(
            "SELECT COUNT(*) FROM notifications WHERE seen = 0 AND user_id = ?",
            (username,)
        ).fetchone()[0]
    conn.close()
    return {"unread": count}


@app.get("/risk_dashboard", response_class=HTMLResponse)
async def risk_dashboard(request: Request):
    """Display categorized student risk levels with scores and recommendations"""
    conn = get_db_connection()
    students_raw = conn.execute("""
        SELECT student_id, risk_level, risk_score, recommendations, predicted_final_grade
        FROM academic_analysis
    """).fetchall()
    conn.close()

    students = []
    for row in students_raw:
        s = dict(row)  # ✅ Convert sqlite3.Row → dict so we can modify it
        try:
            recs = json.loads(s.get("recommendations", "[]"))
            if isinstance(recs, str):
                recs = [recs]
            s["recommendations"] = recs
        except Exception:
            s["recommendations"] = [s.get("recommendations", "No recommendations")]
        students.append(s)

    # ✅ Categorize by risk level
    critical_students = [s for s in students if s["risk_level"] == "CRITICAL"]
    high_students = [s for s in students if s["risk_level"] == "HIGH"]
    medium_students = [s for s in students if s["risk_level"] == "MEDIUM"]
    low_students = [s for s in students if s["risk_level"] == "LOW"]

    # ✅ Render the template
    return templates.TemplateResponse(
        "risk_dashboard.html",
        {
            "request": request,
            "critical_students": critical_students,
            "high_students": high_students,
            "medium_students": medium_students,
            "low_students": low_students,
        },
    )
# ========== RUN APPLICATION ==========

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"

    )
