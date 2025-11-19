# [file name]: agents/core.py
# [file content begin]
# agents/core.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
import json

class BaseAgent(ABC):
    """Enhanced autonomous agent with memory, logging, and learning"""
    
    def __init__(self, name: str):
        self.name = name
        self.memory = AgentMemory()
        self.goals = []
        self.current_plan = None
        self.performance_metrics = {
            "goals_achieved": 0,
            "plans_executed": 0,
            "success_rate": 0.0,
            "actions_logged": 0,
            "memories_stored": 0
        }
        
    async def log_action(self, action_type: str, details: str, student_id: str = None,
                        target_user: str = None, context_before: dict = None, 
                        success_score: float = 0.0, learned_lessons: str = None):
        """Log agent action with context"""
        from main import log_agent_action
        
        context_after = await self.get_current_context()
        
        log_agent_action(
            agent_name=self.name,
            action_type=action_type,
            details=details,
            student_id=student_id,
            target_user=target_user,
            context_before=context_before,
            context_after=context_after,
            success_score=success_score,
            learned_lessons=learned_lessons
        )
        
        self.performance_metrics["actions_logged"] += 1
    
    async def store_memory(self, memory_type: str, content: str, student_id: str = None,
                          context_data: dict = None, confidence_score: float = 1.0):
        """Store agent memory for future learning"""
        from main import store_agent_memory
        
        if not context_data:
            context_data = await self.get_current_context()
        
        store_agent_memory(
            agent_name=self.name,
            memory_type=memory_type,
            content=content,
            student_id=student_id,
            context_data=context_data,
            confidence_score=confidence_score
        )
        
        self.performance_metrics["memories_stored"] += 1
    
    async def get_memories(self, memory_type: str = None, student_id: str = None, limit: int = 10):
        """Retrieve relevant memories"""
        from main import get_agent_memories
        return get_agent_memories(self.name, memory_type, student_id, limit)
    
    async def get_suggestions(self, student_id: str = None):
        """Get suggestions based on past memories"""
        from main import get_suggestions_from_memory
        return get_suggestions_from_memory(self.name, student_id)
    
    async def send_notification(self, message: str, user_id: str = None, 
                              priority: str = "normal", metadata: dict = None):
        """Send targeted notification"""
        from main import add_notification
        
        add_notification(
            message=message,
            user_id=user_id,
            notification_type="agent",
            priority=priority,
            agent_source=self.name,
            metadata=metadata
        )
    
    async def get_current_context(self) -> Dict:
        """Get current agent context for logging"""
        return {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "goals": self.goals,
            "performance_metrics": self.performance_metrics,
            "memory_stats": {
                "success_patterns": len(self.memory.success_patterns),
                "failure_patterns": len(self.memory.failure_patterns),
                "learning_cycles": self.memory.learning_cycles
            }
        }
    
    @abstractmethod
    async def pursue_goals(self, context: Dict) -> Dict:
        """Autonomously pursue agent's goals based on context"""
        pass
    
    @abstractmethod  
    async def create_plan(self, goal: str, context: Dict) -> List[Dict]:
        """Create multi-step plan to achieve goal"""
        pass
    
    async def execute_plan_step(self, step: Dict, context: Dict) -> Dict:
        """Execute a single step in the plan with logging"""
        try:
            # Log action start
            await self.log_action(
                action_type="plan_step",
                details=f"Executing: {step['step']}",
                context_before=context
            )
            
            tool = step.get('tool', 'internal_logic')
            action = step['step']
            
            if tool == 'database':
                result = await self.use_database_tool(action, context)
            elif tool == 'analysis':
                result = await self.perform_analysis(action, context)
            elif tool == 'notification_system':
                result = await self.send_notification_tool(action, context)
            else:
                result = await self.internal_processing(action, context)
            
            # Log successful completion
            await self.log_action(
                action_type="plan_step_completed",
                details=f"Completed: {step['step']}",
                success_score=1.0,
                learned_lessons=f"Successfully executed {action}"
            )
            
            return {"step": action, "status": "completed", "result": result}
            
        except Exception as e:
            # Log failure
            await self.log_action(
                action_type="plan_step_failed",
                details=f"Failed: {step['step']} - Error: {str(e)}",
                success_score=0.0,
                learned_lessons=f"Failed to execute {action}: {str(e)}"
            )
            
            return {"step": action, "status": "failed", "error": str(e)}
    
    async def reflect_on_outcome(self, plan: Dict, outcome: Dict):
        """Enhanced reflection with memory storage"""
        success = outcome.get('success_rate', 0) > 0.6
        
        reflection = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "plan": plan,
            "outcome": outcome,
            "success": success,
            "lessons_learned": await self.analyze_success(plan, outcome),
            "strategy_adjustments": await self.adapt_strategy(plan, outcome)
        }
        
        self.memory.store_reflection(reflection)
        
        # Store in long-term memory
        memory_type = "success_pattern" if success else "failure_pattern"
        memory_content = f"Plan '{plan.get('goal', 'unknown')}' with strategy '{plan.get('strategy', 'default')}' - Success: {success}"
        
        await self.store_memory(
            memory_type=memory_type,
            content=memory_content,
            context_data=reflection,
            confidence_score=outcome.get('success_rate', 0.5)
        )
        
        if success:
            self.performance_metrics["goals_achieved"] += 1
            self.performance_metrics["success_rate"] = (
                self.performance_metrics["goals_achieved"] / 
                max(1, self.performance_metrics["plans_executed"])
            )
        
        return reflection
    
    async def send_agent_notification(self, message: str, user_id: str = None, 
                                        priority: str = "normal", metadata: dict = None):
        """Send notification from this agent"""
        from main import add_notification
        
        add_notification(
            message=message,
            user_id=user_id,
            notification_type="agent",
            priority=priority,
            agent_source=self.name,
            metadata=metadata
        )
        
        

    async def send_notification_tool(self, action: str, context: Dict) -> Dict:
        """Enhanced notification tool with targeting"""
        # Determine notification target and priority
        priority = "normal"
        target_user = None
        
        if "urgent" in action.lower() or "critical" in action.lower():
            priority = "high"
        if "student" in context:
            target_user = context.get("student_id")
        
        message = f"{self.name}: {action}"
        
        
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
    

    async def analyze_success(self, plan: Dict, outcome: Dict) -> List[str]:
        """Analyze what worked and what didn't"""
        lessons = []
        
        if outcome.get('success_rate', 0) > 0.8:
            lessons.append(f"Plan {plan.get('goal', 'unknown')} was highly effective")
            lessons.append(f"Strategy: {plan.get('strategy', 'default')} worked well")
        else:
            lessons.append(f"Plan {plan.get('goal', 'unknown')} needs improvement")
            lessons.append("Consider alternative approaches")
            
        return lessons
    
    async def adapt_strategy(self, plan: Dict, outcome: Dict) -> Dict:
        """Adapt future strategies based on outcomes"""
        adjustments = {}
        
        if outcome.get('success_rate', 0) < 0.5:
            adjustments["change_approach"] = True
            adjustments["try_alternative"] = "escalation_protocol"
            
        return adjustments
    
    # Tool methods
    async def use_database_tool(self, action: str, context: Dict) -> Dict:
        return {"action": action, "status": "tool_not_implemented"}
    
    async def perform_analysis(self, action: str, context: Dict) -> Dict:
        return {"action": action, "status": "analysis_not_implemented"}
    
    async def internal_processing(self, action: str, context: Dict) -> Dict:
        return {"action": action, "status": "processing_completed"}


class AgentMemory:
    """Enhanced agent memory with persistence support"""
    
    def __init__(self):
        self.success_patterns = []
        self.failure_patterns = []
        self.intervention_history = []
        self.learning_cycles = 0
        
    def store_reflection(self, reflection: Dict):
        """Store and learn from reflections"""
        self.intervention_history.append(reflection)
        self.learning_cycles += 1
        
        if reflection["success"]:
            self.success_patterns.append(reflection)
            # Keep only recent successes for adaptive learning
            if len(self.success_patterns) > 50:
                self.success_patterns = self.success_patterns[-25:]
        else:
            self.failure_patterns.append(reflection)
            if len(self.failure_patterns) > 50:
                self.failure_patterns = self.failure_patterns[-25:]
    
    def get_success_patterns(self, goal_type: str) -> List[Dict]:
        """Retrieve successful patterns for specific goals"""
        return [p for p in self.success_patterns 
                if p["plan"].get("goal", "") == goal_type]
    
    def get_recent_lessons(self, count: int = 5) -> List[str]:
        """Get recent lessons learned"""
        recent = self.intervention_history[-count:]
        lessons = []
        for reflection in recent:
            lessons.extend(reflection.get("lessons_learned", []))
        return lessons

# [file content end]
