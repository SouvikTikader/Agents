# agents/academic_agent.py
from .core import BaseAgent, AgentMemory
import asyncio
from typing import Dict, List
from datetime import datetime, timedelta
import sqlite3


class AcademicAgent(BaseAgent):
    """Autonomous academic support agent with predictive analytics"""
    
    def __init__(self):
        super().__init__("AcademicSupportAgent")
        self.operation_cycles = 0
        self.goals = [
            "identify_at_risk_students",
            "improve_academic_performance", 
            "prevent_student_attrition",
            "optimize_learning_paths",
            "predict_future_risks"
        ]
        self.risk_thresholds = {
            "attendance": 60,
            "grades": 40,
            "trend": -10  # Percentage decline
        }
        
    
    async def pursue_goals(self, context: Dict) -> Dict:
        """Autonomously work toward academic goals based on system state"""
        actions_taken = []
        system_context = context.get('academic_context', {})
        
        print(f"🎓 {self.name} pursuing goals: {self.goals}")
        
        for goal in self.goals:
            if await self.should_pursue_goal(goal, system_context):
                print(f"  🎯 Pursuing goal: {goal}")
                
                plan = await self.create_plan(goal, system_context)
                self.performance_metrics["plans_executed"] += 1
                
                # Execute plan steps
                plan_results = []
                for step in plan.get('steps', []):
                    step_result = await self.execute_plan_step(step, system_context)
                    plan_results.append(step_result)
                    await asyncio.sleep(0.1)  # Small delay between steps
                
                # Analyze results
                success_rate = await self.calculate_success_rate(plan_results)
                result = {
                    "goal": goal,
                    "plan": plan,
                    "step_results": plan_results,
                    "success_rate": success_rate,
                    "timestamp": datetime.now().isoformat()
                }
                
                actions_taken.append(result)
                await self.reflect_on_outcome(plan, result)
        
        self.operation_cycles += 1  # INCREMENT OPERATION CYCLES
        return {
            "agent": self.name,
            "actions_taken": actions_taken,
            "goals_pursued": len(actions_taken),
            "timestamp": datetime.now().isoformat()
        }
    
    async def analyze_student_patterns(self, student_id: str, context: Dict) -> Dict:
        """Analyze specific student patterns for cross-agent collaboration"""
        print(f"🎓 Academic Agent analyzing patterns for {student_id}")
        
        # Get student data from context
        student_risk_analysis = context.get('academic_context', {}).get('student_risk_analysis', [])
        student_data = next((s for s in student_risk_analysis if s['student_id'] == student_id), None)
        
        if not student_data:
            return {'risk_level': 'UNKNOWN', 'is_failing': False}
        
        # Analyze academic patterns
        is_failing = student_data.get('risk_level') in ['CRITICAL', 'HIGH']
        trend_analysis = await self.analyze_academic_trend(student_data)
        
        return {
            'student_id': student_id,
            'risk_level': student_data.get('risk_level', 'UNKNOWN'),
            'risk_score': student_data.get('risk_score', 0),
            'is_failing': is_failing,
            'predicted_grade': student_data.get('predicted_final_grade', 0),
            'attendance_issues': self.has_attendance_issues(student_data),
            'performance_trend': trend_analysis,
            'key_issues': self.identify_key_academic_issues(student_data),
            'recommended_academic_actions': student_data.get('recommendations', [])[:3]  # Top 3
        }

    async def analyze_academic_trend(self, student_data: Dict) -> str:
        """Analyze academic performance trend"""
        marks_ta1 = student_data.get('marks_ta1', 0)
        marks_ta2 = student_data.get('marks_ta2', 0)
        
        if marks_ta1 > 0 and marks_ta2 > 0:
            if marks_ta2 < marks_ta1 - 10:
                return "DECLINING"
            elif marks_ta2 > marks_ta1 + 10:
                return "IMPROVING"
            else:
                return "STABLE"
        return "INSUFFICIENT_DATA"

    def has_attendance_issues(self, student_data: Dict) -> bool:
        """Check if student has attendance issues"""
        attendance_fields = ['attendance_before_ta1', 'attendance_before_ta2', 'attendance_before_final']
        return any(student_data.get(field, 100) < 60 for field in attendance_fields)

    def identify_key_academic_issues(self, student_data: Dict) -> List[str]:
        """Identify key academic issues for the student"""
        issues = []
        
        if student_data.get('marks_ta1', 100) < 40:
            issues.append("Failed TA1")
        if student_data.get('marks_ta2', 100) < 40:
            issues.append("Failed TA2") 
        if student_data.get('attendance_before_ta1', 100) < 60:
            issues.append("Low attendance")
        if student_data.get('assignment_marks', 100) < 40:
            issues.append("Poor assignment performance")
            
        return issues
    
    async def calculate_student_risk_score(self, student_data: Dict) -> Dict:
        """Calculate comprehensive risk score for a student"""
        risk_factors = 0
        total_factors = 0
        
        # Academic performance factors (weighted)
        if student_data.get('marks_ta1', 100) < 40: risk_factors += 2
        if student_data.get('marks_ta2', 100) < 40: risk_factors += 2
        if student_data.get('marks_final', 100) < 40: risk_factors += 3
        if student_data.get('assignment_marks', 100) < 40: risk_factors += 1
        
        # Attendance factors
        if student_data.get('attendance_before_ta1', 100) < 60: risk_factors += 1
        if student_data.get('attendance_before_ta2', 100) < 60: risk_factors += 1
        if student_data.get('attendance_before_final', 100) < 60: risk_factors += 2
        
        # Trend analysis
        if (student_data.get('marks_ta1', 0) > 0 and student_data.get('marks_ta2', 0) > 0 and
            student_data.get('marks_ta2', 0) < student_data.get('marks_ta1', 0) - 15):
            risk_factors += 2  # Significant decline
        
        total_factors = 12  # Total possible risk factors
        
        risk_score = risk_factors / total_factors
        
        # Risk levels
        if risk_score >= 0.7: risk_level = "CRITICAL"
        elif risk_score >= 0.5: risk_level = "HIGH"
        elif risk_score >= 0.3: risk_level = "MEDIUM"
        else: risk_level = "LOW"
        
        
        result = {
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommendations": self.generate_risk_recommendations(risk_factors, risk_level),
            "predicted_final_grade": await self.predict_final_grade(student_data)
        }

        # ✅ Save analysis result for future dashboards
        self.save_risk_analysis(student_data.get("student_id", "UNKNOWN"), result)

        return result


    def generate_risk_recommendations(self, risk_factors: int, risk_level: str) -> List[str]:
        """Generate specific recommendations based on risk factors"""
        recommendations = []
        
        if risk_level == "CRITICAL":
            recommendations.extend([
                "🚨 IMMEDIATE: Academic counseling required",
                "🚨 Contact student advisor urgently", 
                "🚨 Consider course withdrawal options",
                "Intensive tutoring program"
            ])
        elif risk_level == "HIGH":
            recommendations.extend([
                "Schedule mandatory tutoring sessions",
                "Weekly progress monitoring", 
                "Parent/guardian notification",
                "Study skills workshop"
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "Recommended study group participation",
                "Office hours attendance",
                "Peer mentoring program",
                "Time management guidance"
            ])
        else:
            recommendations.extend([
                "Maintain current study habits",
                "Regular self-assessment",
                "Participate in class discussions"
            ])
        
        return recommendations

    async def predict_final_grade(self, student_data: Dict) -> float:
        """Predict final grade based on current performance"""
        marks = [
            student_data.get('marks_ta1', 0),
            student_data.get('marks_ta2', 0),
            student_data.get('assignment_marks', 0)
        ]
        
        # Filter out zeros (missing data)
        valid_marks = [m for m in marks if m > 0]
        
        if not valid_marks:
            return 0.0
        
        # Weighted prediction (later assignments matter more)
        if len(valid_marks) >= 2:
            predicted = (valid_marks[0] * 0.3 + valid_marks[1] * 0.4 + 
                        (valid_marks[-1] if len(valid_marks) > 2 else valid_marks[1]) * 0.3)
        else:
            predicted = valid_marks[0]
        
        # Adjust based on attendance trend
        attendance = student_data.get('attendance_before_ta1', 80)
        if attendance < 60:
            predicted *= 0.8  # 20% penalty for low attendance
        elif attendance > 90:
            predicted *= 1.05  # 5% bonus for excellent attendance
            
        return max(0, min(100, predicted))
    
    async def create_plan(self, goal: str, context: Dict) -> Dict:
        """Create intelligent multi-step plans with real actions"""
        
        goal_plans = {
            "identify_at_risk_students": {
                "goal": goal,
                "strategy": "proactive_monitoring",
                "steps": [
                    {"step": "analyze_recent_attendance_trends", "tool": "database"},
                    {"step": "check_assignment_completion_rates", "tool": "database"},
                    {"step": "calculate_individual_risk_scores", "tool": "analysis"},
                    {"step": "flag_high_risk_cases_for_intervention", "tool": "notification_system"},
                    {"step": "create_intervention_plans_for_critical_cases", "tool": "internal_processing"}
                ]
            },
            "improve_academic_performance": {
                "goal": goal,
                "strategy": "targeted_intervention", 
                "steps": [
                    {"step": "identify_common_learning_gaps", "tool": "analysis"},
                    {"step": "analyze_successful_intervention_patterns", "tool": "analysis"},
                    {"step": "create_personalized_study_plans", "tool": "internal_processing"},
                    {"step": "schedule_tutoring_sessions", "tool": "notification_system"},
                    {"step": "monitor_intervention_progress", "tool": "database"}
                ]
            },
            "prevent_student_attrition": {
                "goal": goal,
                "strategy": "retention_focused",
                "steps": [
                    {"step": "monitor_engagement_metrics", "tool": "analysis"},
                    {"step": "identify_disengagement_patterns", "tool": "analysis"},
                    {"step": "initiate_proactive_outreach", "tool": "notification_system"},
                    {"step": "offer_support_services", "tool": "internal_processing"},
                    {"step": "track_intervention_effectiveness", "tool": "analysis"}
                ]
            },
            "optimize_learning_paths": {
                "goal": goal,
                "strategy": "data_driven_optimization",
                "steps": [
                    {"step": "analyze_successful_student_patterns", "tool": "analysis"},
                    {"step": "identify_optimal_learning_sequences", "tool": "analysis"},
                    {"step": "create_personalized_learning_paths", "tool": "internal_processing"},
                    {"step": "implement_adaptive_learning_strategies", "tool": "notification_system"}
                ]
            },
            "predict_future_risks": {
                "goal": goal,
                "strategy": "predictive_analytics",
                "steps": [
                    {"step": "analyze_historical_performance_data", "tool": "database"},
                    {"step": "identify_early_warning_indicators", "tool": "analysis"},
                    {"step": "calculate_future_risk_probabilities", "tool": "analysis"},
                    {"step": "generate_preventive_recommendations", "tool": "notification_system"}
                ]
            }
        }
        
        return goal_plans.get(goal, {"goal": goal, "strategy": "default", "steps": []})
    
    async def should_pursue_goal(self, goal: str, context: Dict) -> bool:
        """Autonomously decide if goal is worth pursuing based on context"""
        
        student_data = context.get('student_performance', {})
        risk_level = context.get('overall_risk_level', 'low')
        recent_alerts = context.get('recent_alerts', [])
        critical_risk = context.get('critical_risk_count', 0)
        high_risk = context.get('high_risk_count', 0)
        
        decision_criteria = {
            "identify_at_risk_students": (
                risk_level in ['high', 'medium'] or 
                len(recent_alerts) > 3 or
                critical_risk > 0 or
                high_risk > 2
            ),
            "improve_academic_performance": (
                risk_level in ['high', 'medium'] or
                context.get('low_performers_count', 0) > 5 or
                len(self.memory.get_recent_lessons(3)) > 0
            ),
            "prevent_student_attrition": (
                risk_level == 'high' or
                context.get('predictive_metrics', {}).get('estimated_dropout_risk', 0) > 0.3 or
                len(recent_alerts) > 5
            ),
            "optimize_learning_paths": (
                len(self.memory.success_patterns) >= 3 or
                context.get('has_sufficient_data', False) or
                self.operation_cycles >= 5
            ),
            "predict_future_risks": (
                critical_risk > 0 or
                context.get('predictive_metrics', {}).get('estimated_dropout_risk', 0) > 0.3 or
                self.operation_cycles >= 2
            )
        }
        
        return decision_criteria.get(goal, False)
    
    async def perform_analysis(self, action: str, context: Dict) -> Dict:
        """Perform academic analysis with predictive notifications"""
        if "risk" in action or "pattern" in action:
            critical_risk = context.get('critical_risk_count', 0)
            high_risk = context.get('high_risk_count', 0)
            dropout_risk = context.get('predictive_metrics', {}).get('estimated_dropout_risk', 0)
            
            print(f"🎓 Predictive Analysis: {critical_risk} critical, {high_risk} high risk, {dropout_risk:.1%} dropout risk")
            
            # Use agent's notification method
            if critical_risk > 0 or high_risk > 2:
                await self.send_agent_notification(
                    message=f"🎓 PREDICTIVE ALERT: {critical_risk} critical risk, {high_risk} high risk students. Dropout risk: {dropout_risk:.1%}",
                    user_id=None,  # Global for admins
                    priority="high",
                    metadata={
                        "critical_risk_count": critical_risk,
                        "high_risk_count": high_risk,
                        "dropout_risk": dropout_risk
                    }
                )
            
            return {
                "action": action, 
                "risk_assessment": "completed", 
                "critical_risk_students": critical_risk,
                "high_risk_students": high_risk,
                "dropout_risk_score": dropout_risk,
                "should_notify": critical_risk > 0 or high_risk > 2
            }
        elif "calculate_individual_risk_scores" in action:
            student_risk_analysis = context.get('student_risk_analysis', [])
            critical_count = len([s for s in student_risk_analysis if s.get('risk_level') == 'CRITICAL'])
            
            if critical_count > 0:
                await self.send_agent_notification(
                    message=f"🔍 Risk Analysis: {len(student_risk_analysis)} students analyzed, {critical_count} critical cases found",
                    user_id=None,  # Global for admins
                    priority="high" if critical_count > 2 else "normal",
                    metadata={
                        "students_analyzed": len(student_risk_analysis),
                        "critical_cases": critical_count
                    }
                )
            
            return {
                "action": action,
                "students_analyzed": len(student_risk_analysis),
                "critical_cases_identified": critical_count,
                "should_notify": critical_count > 0
            }
        elif "gap" in action:
            return {"action": action, "learning_gaps_identified": 2, "recommendations": 3}
        else:
            return {"action": action, "analysis": "completed", "insights": 2}
    
    async def use_database_tool(self, action: str, context: Dict) -> Dict:
        """Use database for academic analysis"""
        if "attendance" in action:
            return {"action": action, "data_analyzed": "attendance_records", "students_checked": 15}
        elif "assignment" in action:
            return {"action": action, "data_analyzed": "assignment_submissions", "completion_rate": "85%"}
        elif "grade" in action:
            return {"action": action, "data_analyzed": "grade_progression", "trends_identified": 3}
        else:
            return {"action": action, "status": "database_query_completed"}
    
    async def send_notification_tool(self, action: str, context: Dict) -> Dict:
        """Send notifications for academic interventions"""
        # Determine notification target and priority
        priority = "normal"
        target_user = None
        
        if "urgent" in action.lower() or "critical" in action.lower():
            priority = "high"
        if "student" in context:
            target_user = context.get("student_id")
        
        message = f"{self.name}: {action}"
        
        # Use agent's notification method
        await self.send_agent_notification(
            message=message,
            user_id=target_user,
            priority=priority,
            metadata={"action": action, "context_keys": list(context.keys())}
        )
        
        return {
            "action": action, 
            "notification_sent": True, 
            "priority": priority,
            "target": target_user or "all"
        }
    
    # Helper methods for decision making
    def has_recent_performance_decline(self, student_data: Dict) -> bool:
        return student_data.get('performance_trend', 'stable') == 'declining'
    
    def has_low_performers(self, student_data: Dict) -> bool:
        return student_data.get('low_performers_count', 0) > 0
    
    def has_attrition_indicators(self, student_data: Dict) -> bool:
        indicators = student_data.get('attrition_indicators', [])
        return len(indicators) >= 2
    
    async def calculate_success_rate(self, plan_results: List[Dict]) -> float:
        """Calculate how successful the plan execution was"""
        if not plan_results:
            return 0.0
        
        completed_steps = sum(1 for r in plan_results if r.get('status') == 'completed')
        return completed_steps / len(plan_results)

    async def generate_predictions(self, student_data: Dict) -> Dict:
        """Generate AI predictions based on student data"""
        # Simple prediction logic - can be enhanced with ML models
        marks = [
            student_data.get('ta1_marks', 0),
            student_data.get('ta2_marks', 0), 
            student_data.get('final_marks', 0),
            student_data.get('assignment_marks', 0)
        ]
        
        avg_marks = sum(marks) / len(marks) if marks else 0
        attendance = student_data.get('attendance', 0)
        
        # Risk assessment
        if avg_marks < 40 or attendance < 60:
            risk_level = "High"
        elif avg_marks < 60 or attendance < 75:
            risk_level = "Medium" 
        else:
            risk_level = "Low"
        
        # Recommendations
        recommendations = []
        if avg_marks < 50:
            recommendations.append("Focus on core concepts")
        if attendance < 75:
            recommendations.append("Improve class attendance")
        if not recommendations:
            recommendations.append("Maintain current performance")
        
        return {
            "predicted_grade": round(avg_marks, 1),
            "risk_level": risk_level,
            "recommendations": recommendations
        }
    def save_risk_analysis(self, student_id: str, risk_data: Dict):
        """Persist academic analysis results to the database"""
        conn = sqlite3.connect('database/system.db')
        conn.execute("""
            CREATE TABLE IF NOT EXISTS academic_analysis (
                student_id TEXT PRIMARY KEY,
                risk_level TEXT,
                risk_score REAL,
                recommendations TEXT,
                predicted_final_grade REAL
            )
        """)

        conn.execute("""
            INSERT OR REPLACE INTO academic_analysis
            (student_id, risk_level, risk_score, recommendations, predicted_final_grade)
            VALUES (?, ?, ?, ?, ?)
        """, (
            student_id,
            risk_data.get("risk_level"),
            risk_data.get("risk_score"),
            "; ".join(risk_data.get("recommendations", [])),
            risk_data.get("predicted_final_grade", 0)
        ))
        conn.commit()
        conn.close()
