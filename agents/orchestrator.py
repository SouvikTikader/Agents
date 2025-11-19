# agents/orchestrator.py
from .academic_agent import AcademicAgent
from .grievance_agent import GrievanceAgent
import asyncio
from typing import Dict, List
from datetime import datetime
import sqlite3
from pathlib import Path

class AgentOrchestrator:
    """Coordinates multiple autonomous agents and provides system context"""
    
    def __init__(self, db_path: str = "database/system.db"):
        self.agents = {
            'academic': AcademicAgent(),
            'grievance': GrievanceAgent()
        }
        self.db_path = Path(db_path)
        self.active = False
        self.operation_cycles = 0
        

    async def start_autonomous_operation(self):
        """Start agents running autonomously in continuous cycles"""
        self.active = True
        print("🤖 Starting autonomous agent system...")
        print("Agents activated:", list(self.agents.keys()))
        
        while self.active:
            try:
                self.operation_cycles += 1
                print(f"\n--- Agent Cycle #{self.operation_cycles} ---")
                
                # Gather current system state
                context = await self.gather_system_context()
                
        
                await self.facilitate_agent_communication(context)
                
                # Let each agent pursue their goals autonomously
                agent_results = []
                for agent_name, agent in self.agents.items():
                    try:
                        print(f"\n🔄 Activating {agent_name}...")
                        result = await agent.pursue_goals(context)
                        agent_results.append(result)
                        print(f"✅ {agent_name} completed with {len(result.get('actions_taken', []))} actions")
                    except Exception as e:
                        print(f"❌ {agent_name} error: {e}")
                        continue
                
                # Process results and trigger notifications
                await self.process_agent_results(agent_results)
                
                # Log and learn from this cycle
                await self.log_agent_activities(agent_results)
                await self.analyze_system_health(agent_results)
                
                print(f"🎯 Cycle #{self.operation_cycles} completed. Waiting 2 minutes...")
                await asyncio.sleep(120)  # 2 minutes between cycles
                
            except Exception as e:
                print(f"❌ Agent system error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error    
    async def gather_system_context(self) -> Dict:
        """Collect real-time data from database for agents to make decisions"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            # Get academic performance data
            performance_data = conn.execute('''
                SELECT COUNT(*) as total_students,
                    AVG(attendance_before_ta1) as avg_attendance,
                    AVG(marks_ta1) as avg_marks_ta1,
                    AVG(marks_ta2) as avg_marks_ta2
                FROM performance
            ''').fetchone()
            
            # Get actual at-risk students count
            at_risk_count = await self.count_at_risk_students()
            
            # Get ALL individual student data for detailed analysis
            all_students = conn.execute('''
                SELECT student_id, marks_ta1, marks_ta2, marks_final, assignment_marks,
                    attendance_before_ta1, attendance_before_ta2, attendance_before_final
                FROM performance
            ''').fetchall()

            # Get individual student risk analysis
            student_risk_analysis = []
            academic_agent = self.agents['academic']
            
            for student in all_students:
                student_dict = dict(student)
                risk_analysis = await academic_agent.calculate_student_risk_score(student_dict)

                #Ensure valid dictionary before unpacking
                if isinstance(risk_analysis, dict):
                    student_risk_analysis.append({
                        "student_id": student_dict['student_id'],
                        **risk_analysis
                    })
                else:
                    print(f"⚠️ Invalid risk_analysis for {student_dict['student_id']}: {risk_analysis}")
                    continue

                
            
            # Count by risk level
            critical_risk = len([s for s in student_risk_analysis if s['risk_level'] == 'CRITICAL'])
            high_risk = len([s for s in student_risk_analysis if s['risk_level'] == 'HIGH'])
            medium_risk = len([s for s in student_risk_analysis if s['risk_level'] == 'MEDIUM'])
            
            # Get grievance data
            grievance_data = conn.execute('''
                SELECT COUNT(*) as total_grievances,
                    SUM(CASE WHEN status = 'Pending' THEN 1 ELSE 0 END) as pending_grievances,
                    SUM(CASE WHEN priority = 'High' THEN 1 ELSE 0 END) as high_priority_grievances
                FROM grievances
            ''').fetchone()
            
            # Get actual pending grievances for detailed processing
            actual_pending_grievances = conn.execute('''
                SELECT id, student_id, category, priority, status, message
                FROM grievances 
                WHERE status = 'Pending'
                LIMIT 20
            ''').fetchall()
            
            conn.close()
            
            # Convert to proper format
            pending_grievances_list = [dict(row) for row in actual_pending_grievances]
            
            # Calculate risk levels
            overall_risk_level = await self.calculate_risk_level(performance_data, grievance_data)
            
            print(f"📊 Enhanced Context: {critical_risk} critical, {high_risk} high, {medium_risk} medium risk students")
            
            return {
                "academic_context": {
                    "student_performance": dict(performance_data) if performance_data else {},
                    "student_risk_analysis": student_risk_analysis,  # NEW
                    "critical_risk_count": critical_risk,  # NEW
                    "high_risk_count": high_risk,  # NEW  
                    "medium_risk_count": medium_risk,  # NEW
                    "overall_risk_level": overall_risk_level,
                    "at_risk_students_count": at_risk_count,
                    "performance_trend": await self.analyze_performance_trend(performance_data),
                    "low_performers_count": await self.count_low_performers(performance_data),
                    "predictive_metrics": {  # NEW
                        "estimated_dropout_risk": await self.calculate_dropout_risk(student_risk_analysis),
                        "intervention_priority_level": await self.calculate_intervention_priority(student_risk_analysis),
                        "system_health_score": await self.calculate_system_health(student_risk_analysis, grievance_data)
                    }
                },
                "grievance_context": {
                    "pending_grievances": pending_grievances_list,
                    "total_grievances": grievance_data['total_grievances'] if grievance_data else 0,
                    "high_priority_count": grievance_data['high_priority_grievances'] if grievance_data else 0,
                    "urgency_metrics": {
                        "high_priority_count": grievance_data['high_priority_grievances'] if grievance_data else 0,
                        "escalation_risk_score": await self.calculate_escalation_risk(grievance_data)
                    }
                },
                "system_health": {
                    "database_connected": True,
                    "agent_count": len(self.agents),
                    "cycle_number": self.operation_cycles,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"Database context gathering error: {e}")
            return 
    async def calculate_dropout_risk(self, risk_analysis: List[Dict]) -> float:
        """Calculate overall system dropout risk"""
        if not risk_analysis:
            return 0.0
        
        high_risk_students = len([s for s in risk_analysis if s['risk_level'] in ['CRITICAL', 'HIGH']])
        total_students = len(risk_analysis)
        
        dropout_risk = min(1.0, (high_risk_students / total_students) * 1.5)  # Scale factor
        
        print(f"📉 Dropout Risk Calculation: {high_risk_students}/{total_students} = {dropout_risk:.1%}")
        return dropout_risk

    async def calculate_intervention_priority(self, risk_analysis: List[Dict]) -> str:
        """Calculate overall intervention priority"""
        critical_count = len([s for s in risk_analysis if s['risk_level'] == 'CRITICAL'])
        high_count = len([s for s in risk_analysis if s['risk_level'] == 'HIGH'])
        
        if critical_count > 3: 
            priority = "CRITICAL"
        elif critical_count > 0 or high_count > 5: 
            priority = "HIGH"
        elif high_count > 2: 
            priority = "MEDIUM"
        else: 
            priority = "NORMAL"
        
        print(f"🎯 Intervention Priority: {priority} ({critical_count} critical, {high_count} high)")
        return priority

    async def calculate_system_health(self, risk_analysis: List[Dict], grievance_data) -> float:
        """Calculate overall system health score"""
        if not risk_analysis:
            return 0.0
        
        
        critical_ratio = len([s for s in risk_analysis if s['risk_level'] == 'CRITICAL']) / len(risk_analysis)
        high_ratio = len([s for s in risk_analysis if s['risk_level'] == 'HIGH']) / len(risk_analysis)
        
        health_score = 1.0 - (critical_ratio * 0.7 + high_ratio * 0.3)
        
        # Adjust for grievance load
        if grievance_data and grievance_data['pending_grievances'] > 10:
            health_score *= 0.8
        
        return max(0.0, min(1.0, health_score))
        
    async def count_at_risk_students(self) -> int:
        """Count actual at-risk students from database with better criteria"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            
            count = conn.execute('''
                SELECT COUNT(*) FROM performance 
                WHERE (marks_ta1 < 40 OR marks_ta2 < 40 OR marks_final < 40 OR 
                    assignment_marks < 40 OR
                    attendance_before_ta1 < 60 OR attendance_before_ta2 < 60 OR 
                    attendance_before_final < 60)
            ''').fetchone()[0]
            
            conn.close()
            print(f"📊 Found {count} at-risk students in database")
            return count
        except Exception as e:
            print(f"Error counting at-risk students: {e}")
            return 0

    #  helper method
    
    async def calculate_risk_level(self, performance_data, grievance_data) -> str:
        """Calculate overall system risk level"""
        if not performance_data:
            return "unknown"
            
        risk_factors = 0
        
        # Academic risk factors
        if performance_data['avg_attendance'] and performance_data['avg_attendance'] < 70:
            risk_factors += 1
        if performance_data['avg_marks_ta1'] and performance_data['avg_marks_ta1'] < 50:
            risk_factors += 1
        if performance_data['avg_marks_ta2'] and performance_data['avg_marks_ta2'] < 50:
            risk_factors += 1
            
        # Grievance risk factors
        if grievance_data and grievance_data['pending_grievances'] > 5:
            risk_factors += 1
        if grievance_data and grievance_data['high_priority_grievances'] > 2:
            risk_factors += 1
            
        if risk_factors >= 3:
            return "high"
        elif risk_factors >= 2:
            return "medium"
        else:
            return "low"
    
    async def analyze_performance_trend(self, performance_data) -> str:
        """Analyze if performance is improving or declining"""
        if not performance_data or not performance_data['avg_marks_ta1'] or not performance_data['avg_marks_ta2']:
            return "stable"
            
        if performance_data['avg_marks_ta2'] > performance_data['avg_marks_ta1']:
            return "improving"
        elif performance_data['avg_marks_ta2'] < performance_data['avg_marks_ta1']:
            return "declining"
        else:
            return "stable"
    async def process_agent_results(self, agent_results: List[Dict]):
        """Process agent results and trigger notifications"""
        from main import add_notification          
        for result in agent_results:
            agent_name = result.get('agent', 'Unknown')
            actions = result.get('actions_taken', [])
            
            for action in actions:
                step_results = action.get('step_results', [])
                for step_result in step_results:
                    step_data = step_result.get('result', {})
                    
                    # Check if this step should trigger a notification
                    if step_data.get('should_notify'):
                        message = step_data.get('notification_message', 
                                            f"{agent_name} completed action: {step_result.get('step')}")
                        add_notification(message)
                        print(f"📢 Agent Notification: {message}")
    
    async def count_low_performers(self, performance_data) -> int:
        """Estimate number of low performers"""
       
        if not performance_data or not performance_data['total_students']:
            return 0
        return max(1, int(performance_data['total_students'] * 0.2))  # Estimate 20%
    
    async def calculate_escalation_risk(self, grievance_data) -> float:
        """Calculate risk of grievance escalation"""
        if not grievance_data or grievance_data['total_grievances'] == 0:
            return 0.0
            
        pending_ratio = grievance_data['pending_grievances'] / grievance_data['total_grievances']
        high_priority_ratio = grievance_data['high_priority_grievances'] / max(1, grievance_data['total_grievances'])
        
        return min(1.0, (pending_ratio * 0.6 + high_priority_ratio * 0.4))
    
    def get_fallback_context(self) -> Dict:
        """Provide fallback context when database is unavailable"""
        return {
            "academic_context": {
                "student_performance": {"total_students": 0, "avg_attendance": 0, "avg_marks_ta1": 0, "avg_marks_ta2": 0},
                "overall_risk_level": "unknown",
                "at_risk_students_count": 0,
                "performance_trend": "unknown",
                "low_performers_count": 0
            },
            "grievance_context": {
                "pending_grievances": 0,
                "total_grievances": 0,
                "high_priority_count": 0,
                "urgency_metrics": {
                    "high_priority_count": 0,
                    "escalation_risk_score": 0.0
                }
            },
            "system_health": {
                "database_connected": False,
                "agent_count": len(self.agents),
                "cycle_number": self.operation_cycles,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    async def log_agent_activities(self, agent_results: List[Dict]):
        """Log what agents accomplished in this cycle"""
        print(f"\n📊 Agent Activity Log - Cycle #{self.operation_cycles}")
        for result in agent_results:
            agent_name = result.get('agent', 'Unknown')
            actions = result.get('actions_taken', [])
            print(f"  {agent_name}: {len(actions)} goals pursued")
            for action in actions:
                print(f"    - {action['goal']} (success: {action['success_rate']:.1%})")
    
    async def analyze_system_health(self, agent_results: List[Dict]):
        """Analyze overall system health and agent performance"""
        total_actions = sum(len(result.get('actions_taken', [])) for result in agent_results)
        avg_success = 0.0
        success_rates = []
        
        for result in agent_results:
            for action in result.get('actions_taken', []):
                success_rates.append(action.get('success_rate', 0.0))
        
        if success_rates:
            avg_success = sum(success_rates) / len(success_rates)
        
        print(f"🏥 System Health: {total_actions} total actions, {avg_success:.1%} average success rate")
        
        # Adaptive learning: Adjust agent behavior based on performance
        if avg_success < 0.5 and self.operation_cycles > 3:
            print("   ⚠️  System performance low - agents will adapt strategies")
    
    def stop_autonomous_operation(self):
        """Stop the autonomous agent system"""
        self.active = False
        print("🛑 Autonomous agent system stopped")
    async def get_student_interventions(self, student_id: str) -> List[Dict]:
        """Get interventions for a specific student"""
        interventions = []
        
        # Check academic agent memory for interventions related to this student
        academic_agent = self.agents['academic']
        for reflection in academic_agent.memory.intervention_history:
            if student_id in str(reflection):  # Simple check for student relevance
                interventions.append({
                    "student_id": student_id,
                    "agent": "AcademicAgent",
                    "intervention_type": reflection.get("plan", {}).get("goal", "unknown"),
                    "timestamp": reflection.get("timestamp"),
                    "success": reflection.get("success", False),
                    "message": f"Academic intervention for {reflection.get('plan', {}).get('goal', 'goal')}",
                    "suggested_action": "Review academic performance and attendance"
                })
        
        return interventions

    async def get_recent_interventions(self) -> List[Dict]:
        """Get recent interventions from all agents"""
        interventions = []
        
        for agent_name, agent in self.agents.items():
            recent_reflections = agent.memory.intervention_history[-10:]  # Last 10
            for reflection in recent_reflections:
                interventions.append({
                    "agent": agent_name,
                    "student_id": "multiple",  # Could be enhanced to track specific students
                    "intervention_type": reflection.get("plan", {}).get("goal", "unknown"),
                    "timestamp": reflection.get("timestamp"),
                    "success": reflection.get("success", False)
                })
        
        return interventions
    
    async def facilitate_agent_communication(self, context: Dict):
        """Facilitate communication between agents based on shared context"""
        academic_context = context.get('academic_context', {})
        grievance_context = context.get('grievance_context', {})
        
        # Find students who are both at-risk AND have pending grievances
        at_risk_students = [s['student_id'] for s in academic_context.get('student_risk_analysis', []) 
                        if s['risk_level'] in ['CRITICAL', 'HIGH']]
        
        students_with_grievances = [g['student_id'] for g in grievance_context.get('pending_grievances', [])]
        
        # Find intersection - students with both academic risk and grievances
        high_risk_with_issues = set(at_risk_students) & set(students_with_grievances)
        
        if high_risk_with_issues:
            
            from main import add_notification
            message = f"🚨 AGENT COLLABORATION: {len(high_risk_with_issues)} high-risk students also have pending grievances"
            
            add_notification(
                message=message,
                user_id=None,  # Global for admins
                notification_type="agent_collaboration",
                priority="high",
                agent_source="Orchestrator",
                metadata={"critical_students": list(high_risk_with_issues)}
            )
            print(f"🤝 {message}")
            
            # Store for coordinated intervention
            self.high_priority_cases = list(high_risk_with_issues)
            
            # Trigger cross-agent analysis
            await self.trigger_cross_agent_analysis(high_risk_with_issues, context)

    async def trigger_cross_agent_analysis(self, critical_students: set, context: Dict):
        """Trigger coordinated analysis for critical cases"""
        print(f"🔍 Cross-agent analysis triggered for {len(critical_students)} critical students")
        
        for student_id in list(critical_students)[:5]:  # Limit to 5 for performance
            # Academic agent analyzes academic patterns
            academic_insights = await self.agents['academic'].analyze_student_patterns(student_id, context)
            
            # Grievance agent analyzes grievance patterns  
            grievance_insights = await self.agents['grievance'].analyze_student_grievance_patterns(student_id, context)
            
            # Combine insights for coordinated intervention
            combined_insights = await self.synthesize_cross_agent_insights(
                student_id, academic_insights, grievance_insights
            )
            
            if combined_insights.get('requires_immediate_action'):
                from main import add_notification
                add_notification(
                    message=f"🎯 COORDINATED INTERVENTION: {student_id} - {combined_insights.get('action_plan', 'Immediate support needed')}",
                    user_id=None,  # Global for admins
                    notification_type="intervention_alert",
                    priority="high",
                    agent_source="Orchestrator",
                    metadata={
                        "student_id": student_id,
                        "intervention_type": combined_insights.get('intervention_type'),
                        "action_plan": combined_insights.get('action_plan')
                    }
                )

    async def synthesize_cross_agent_insights(self, student_id: str, academic_insights: Dict, grievance_insights: Dict) -> Dict:
        """Synthesize insights from both agents into coordinated action plan"""
        
        action_priority = "HIGH" if academic_insights.get('risk_level') in ['CRITICAL', 'HIGH'] else "MEDIUM"
        
        # Determine the most appropriate intervention type
        if grievance_insights.get('has_urgent_grievances') and academic_insights.get('is_failing'):
            intervention_type = "ACADEMIC_COUNSELING_AND_GRIEVANCE_RESOLUTION"
            action_plan = "Schedule emergency meeting with academic advisor and grievance officer"
        elif academic_insights.get('is_failing'):
            intervention_type = "ACADEMIC_EMERGENCY" 
            action_plan = "Immediate academic counseling and tutoring"
        elif grievance_insights.get('has_urgent_grievances'):
            intervention_type = "GRIEVANCE_URGENT"
            action_plan = "Expedited grievance resolution with counseling support"
        else:
            intervention_type = "MONITORING"
            action_plan = "Enhanced monitoring and regular check-ins"
        
        return {
            'student_id': student_id,
            'intervention_type': intervention_type,
            'action_priority': action_priority,
            'action_plan': action_plan,
            'requires_immediate_action': action_priority == "HIGH",
            'academic_risk': academic_insights.get('risk_level', 'UNKNOWN'),
            'grievance_urgency': grievance_insights.get('urgency_level', 'UNKNOWN'),
            'combined_risk_score': self.calculate_combined_risk_score(academic_insights, grievance_insights),
            'timestamp': datetime.now().isoformat()
        }

    def calculate_combined_risk_score(self, academic_insights: Dict, grievance_insights: Dict) -> float:
        """Calculate combined risk score from both academic and grievance perspectives"""
        academic_risk = {
            'CRITICAL': 1.0, 'HIGH': 0.7, 'MEDIUM': 0.4, 'LOW': 0.1
        }.get(academic_insights.get('risk_level', 'LOW'), 0.1)
        
        grievance_risk = {
            'URGENT': 0.8, 'HIGH': 0.6, 'MEDIUM': 0.3, 'LOW': 0.1
        }.get(grievance_insights.get('urgency_level', 'LOW'), 0.1)
        
        # Weight academic risk
        return (academic_risk * 0.6 + grievance_risk * 0.4)

    async def count_low_performers(self, performance_data) -> int:
        """Count actual low performers based on risk analysis"""
        if not hasattr(performance_data, 'get'):
            return 0
        return len([s for s in performance_data.get('student_risk_analysis', []) 
                    if s.get('risk_level') in ['CRITICAL', 'HIGH']])
    
    async def get_agent_status(self) -> Dict:
        """Get current status of all agents"""
        status = {
            "system_active": self.active,
            "operation_cycles": self.operation_cycles,
            "agents": {}
        }
        
    
        
        for agent_name, agent in self.agents.items():
            status["agents"][agent_name] = {
                "goals": agent.goals,
                "performance_metrics": agent.performance_metrics,
                "memory_usage": {
                    "success_patterns": len(agent.memory.success_patterns),
                    "failure_patterns": len(agent.memory.failure_patterns),
                    "learning_cycles": agent.memory.learning_cycles
                }
            }
        

        return status
