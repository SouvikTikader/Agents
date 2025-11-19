# agents/grievance_agent.py
from .core import BaseAgent, AgentMemory
import asyncio
from typing import Dict, List
from datetime import datetime

class GrievanceAgent(BaseAgent):
    """Autonomous grievance resolution agent with escalation intelligence"""
    
    def __init__(self):
        super().__init__("GrievanceResolutionAgent")
        self.operation_cycles = 0
        self.goals = [
            "resolve_grievances_quickly",
            "prevent_issue_escalation", 
            "identify_systemic_problems",
            "improve_response_times"
        ]
        self.escalation_thresholds = {
            "pending_time_hours": 24,
            "priority_high_count": 3,
            "same_student_repeats": 2
        }
        
    
    async def analyze_student_grievance_patterns(self, student_id: str, context: Dict) -> Dict:
        """Analyze grievance patterns for specific student for cross-agent collaboration"""
        print(f"⚖️ Grievance Agent analyzing patterns for {student_id}")
        
        pending_grievances = context.get('grievance_context', {}).get('pending_grievances', [])
        student_grievances = [g for g in pending_grievances if g['student_id'] == student_id]
        
        if not student_grievances:
            return {'has_urgent_grievances': False, 'urgency_level': 'LOW'}
        
        # Analyze grievance patterns
        urgent_count = len([g for g in student_grievances if g['priority'] == 'High'])
        categories = [g['category'] for g in student_grievances]
        category_counts = {category: categories.count(category) for category in set(categories)}
        
        # Determine urgency level
        if urgent_count > 0:
            urgency_level = 'URGENT'
        elif len(student_grievances) >= 3:
            urgency_level = 'HIGH' 
        elif len(student_grievances) >= 2:
            urgency_level = 'MEDIUM'
        else:
            urgency_level = 'LOW'
        
        return {
            'student_id': student_id,
            'total_grievances': len(student_grievances),
            'urgent_grievances': urgent_count,
            'has_urgent_grievances': urgent_count > 0,
            'urgency_level': urgency_level,
            'main_categories': dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:2]),
            'recurring_issues': self.identify_recurring_issues(student_grievances),
            'suggested_grievance_actions': self.suggest_grievance_actions(student_grievances)
        }

    def identify_recurring_issues(self, grievances: List[Dict]) -> List[str]:
        """Identify recurring issues in student grievances"""
        categories = [g['category'] for g in grievances]
        category_counts = {category: categories.count(category) for category in set(categories)}
        
        recurring = []
        for category, count in category_counts.items():
            if count >= 2:
                recurring.append(f"{category} ({count} times)")
        
        return recurring

    def suggest_grievance_actions(self, grievances: List[Dict]) -> List[str]:
        """Suggest actions based on grievance patterns"""
        actions = []
        categories = [g['category'] for g in grievances]
        
        if 'academic' in categories:
            actions.append("Coordinate with academic advisor")
        if 'faculty' in categories:
            actions.append("Schedule meeting with department head")
        if len(grievances) >= 3:
            actions.append("Comprehensive student support review")
        if any(g['priority'] == 'High' for g in grievances):
            actions.append("Immediate grievance resolution")
        
        return actions if actions else ["Standard grievance processing"]
    
    async def pursue_goals(self, context: Dict) -> Dict:
        """Autonomously handle grievance-related goals"""
        actions_taken = []
        grievance_context = context.get('grievance_context', {})
        
        # Use agent's notification method instead of direct import
        pending_count = len(grievance_context.get('pending_grievances', []))
        if pending_count > 0:
            await self.send_agent_notification(
                message=f"⚖️ Grievance Agent: Processing {pending_count} pending grievances",
                priority="normal",
                metadata={"pending_grievances": pending_count}
            )
        
        # Handle case where pending_grievances might be an integer
        pending_grievances = grievance_context.get('pending_grievances', [])
        if isinstance(pending_grievances, int):
            # If it's a count, create a mock list for processing
            pending_grievances = [{"id": i, "category": "general", "priority": "medium"} 
                                for i in range(min(pending_grievances, 10))]
        
        print(f"⚖️ {self.name} monitoring {len(pending_grievances)} pending grievances")

        for goal in self.goals:
            if await self.should_pursue_goal(goal, grievance_context):
                print(f"  🎯 Pursuing grievance goal: {goal}")
                
                plan = await self.create_plan(goal, grievance_context)
                self.performance_metrics["plans_executed"] += 1
                
                # Execute plan
                plan_results = []
                for step in plan.get('steps', []):
                    step_result = await self.execute_plan_step(step, grievance_context)
                    plan_results.append(step_result)
                    await asyncio.sleep(0.1)
                
                # Calculate success
                success_rate = await self.calculate_grievance_success(plan_results, grievance_context)
                result = {
                    "goal": goal,
                    "plan": plan,
                    "step_results": plan_results,
                    "success_rate": success_rate,
                    "grievances_processed": len(pending_grievances),
                    "timestamp": datetime.now().isoformat()
                }
                
                actions_taken.append(result)
                await self.reflect_on_outcome(plan, result)
        
        self.operation_cycles += 1  # INCREMENT OPERATION CYCLES
        return {
            "agent": self.name,
            "actions_taken": actions_taken,
            "goals_pursued": len(actions_taken),
            "grievances_monitored": len(pending_grievances),
            "timestamp": datetime.now().isoformat()
        }
    
    async def create_plan(self, goal: str, context: Dict) -> Dict:
        """Create dynamic grievance resolution plans with real actions"""
        
        goal_plans = {
            "resolve_grievances_quickly": {
                "goal": goal,
                "strategy": "efficient_triage",
                "steps": [
                    {"step": "categorize_pending_grievances", "tool": "analysis"},
                    {"step": "prioritize_by_urgency_and_impact", "tool": "internal_logic"},
                    {"step": "assign_appropriate_resolution_paths", "tool": "internal_logic"},
                    {"step": "initiate_quick_resolution_protocols", "tool": "notification_system"},
                    {"step": "track_resolution_progress", "tool": "database"}
                ]
            },
            "prevent_issue_escalation": {
                "goal": goal,
                "strategy": "proactive_containment",
                "steps": [
                    {"step": "identify_potential_escalation_risks", "tool": "analysis"},
                    {"step": "analyze_escalation_patterns_from_history", "tool": "analysis"},
                    {"step": "implement_early_intervention_measures", "tool": "notification_system"},
                    {"step": "monitor_intervention_effectiveness", "tool": "analysis"},
                    {"step": "adjust_containment_strategy_if_needed", "tool": "internal_logic"}
                ]
            },
            "identify_systemic_problems": {
                "goal": goal,
                "strategy": "pattern_analysis",
                "steps": [
                    {"step": "analyze_grievance_trends_over_time", "tool": "analysis"},
                    {"step": "identify_recurring_issue_patterns", "tool": "analysis"},
                    {"step": "correlate_with_academic_performance_data", "tool": "database"},
                    {"step": "flag_systemic_issues_for_review", "tool": "notification_system"},
                    {"step": "recommend_system_improvements", "tool": "internal_logic"}
                ]
            },
            "improve_response_times": {
                "goal": goal,
                "strategy": "performance_optimization",
                "steps": [
                    {"step": "analyze_current_response_time_metrics", "tool": "analysis"},
                    {"step": "identify_bottlenecks_in_resolution_process", "tool": "analysis"},
                    {"step": "implement_process_improvements", "tool": "internal_logic"},
                    {"step": "monitor_improvement_impact", "tool": "analysis"},
                    {"step": "adjust_strategy_based_on_results", "tool": "internal_logic"}
                ]
            }
        }
        
        return goal_plans.get(goal, {"goal": goal, "strategy": "default", "steps": []})
    
    async def should_pursue_goal(self, goal: str, context: Dict) -> bool:
        """Autonomously decide when to pursue grievance goals"""
        
        pending_grievances = context.get('pending_grievances', [])
        urgency_metrics = context.get('urgency_metrics', {})
        recent_escalations = context.get('recent_escalations', 0)
        
        decision_criteria = {
            "resolve_grievances_quickly": (
                len(pending_grievances) >= 5 or
                urgency_metrics.get('high_priority_count', 0) >= 2 or
                urgency_metrics.get('avg_pending_hours', 0) > 12
            ),
            "prevent_issue_escalation": (
                recent_escalations >= 1 or
                urgency_metrics.get('escalation_risk_score', 0) > 0.7 or
                self.has_repeating_issues(pending_grievances)
            ),
            "identify_systemic_problems": (
                len(pending_grievances) >= 10 or
                self.has_common_patterns(pending_grievances) or
                self.operation_cycles >= 3  # Run after some cycles
            ),
            "improve_response_times": (
                urgency_metrics.get('avg_response_time_hours', 0) > 24 or
                self.memory.learning_cycles >= 5
            )
        }
        
        return decision_criteria.get(goal, False)
    
    async def use_database_tool(self, action: str, context: Dict) -> Dict:
        """Use database for grievance analysis"""
        if "categorize" in action:
            return {"action": action, "grievances_categorized": 8, "categories_used": 4}
        elif "track" in action:
            return {"action": action, "progress_tracked": True, "resolution_rate": "75%"}
        elif "correlate" in action:
            return {"action": action, "data_correlated": True, "insights_found": 2}
        else:
            return {"action": action, "database_operation": "completed"}
    
    async def perform_analysis(self, action: str, context: Dict) -> Dict:
        """Perform grievance pattern analysis"""
        if "prioritize" in action:
            return {"action": action, "grievances_prioritized": 5, "high_priority": 2}
        elif "escalation" in action:
            return {"action": action, "risks_identified": 3, "preventive_actions": 2}
        elif "pattern" in action or "trend" in action:
            # Send notification for systemic issues
            if "systemic" in action.lower():
                await self.send_agent_notification(
                    message=f"🔍 Systemic patterns identified in grievances",
                    priority="normal",
                    metadata={"action": action, "patterns_found": 4}
                )
            return {"action": action, "patterns_identified": 4, "systemic_issues": 1}
        else:
            return {"action": action, "analysis_completed": True}
    
    async def send_notification_tool(self, action: str, context: Dict) -> Dict:
        """Send grievance-related notifications with enhanced targeting"""
        if "initiate" in action:
            await self.send_agent_notification(
                message=f"⚖️ Grievance protocols activated for urgent cases",
                priority="high",
                metadata={"action": action, "teams": ["support", "academic"]}
            )
            
            return {
                "action": action, 
                "protocols_activated": 2, 
                "teams_notified": ["support", "academic"],
                "notification_sent": True
            }
        elif "flag" in action:
            await self.send_agent_notification(
                message=f"🚩 Systemic issues flagged for administrative review",
                priority="normal",
                metadata={"action": action, "reviewers": ["admin"]}
            )
            
            return {
                "action": action, 
                "issues_flagged": 1, 
                "reviewers_notified": ["admin"],
                "notification_sent": True
            }
        else:
            await self.send_agent_notification(
                message=f"⚖️ Grievance system update: {action}",
                priority="normal"
            )
            
            return {
                "action": action, 
                "notification_sent": True, 
                "type": "grievance_alert"
            }
    
    # Helper methods
    def has_repeating_issues(self, grievances: List[Dict]) -> bool:
        """Check if there are repeating issues from same students"""
        student_issues = {}
        for grievance in grievances:
            student_id = grievance.get('student_id')
            if student_id in student_issues:
                return True
            student_issues[student_id] = True
        return False
    
    def has_common_patterns(self, grievances: List[Dict]) -> bool:
        """Check for common patterns across grievances"""
        categories = [g.get('category') for g in grievances]
        return len(set(categories)) < len(categories)  # If duplicates exist
    
    async def calculate_grievance_success(self, plan_results: List[Dict], context: Dict) -> float:
        """Calculate success rate for grievance resolution plans"""
        if not plan_results:
            return 0.0
        
        successful_steps = sum(1 for r in plan_results 
                             if r.get('status') == 'completed' and 
                             not r.get('result', {}).get('error'))
        return successful_steps / len(plan_results)