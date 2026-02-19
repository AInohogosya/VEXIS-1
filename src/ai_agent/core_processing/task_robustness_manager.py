"""
Task Robustness Manager for AI Agent System
Ensures tasks execute completely through all steps without premature termination
"""

import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logger import get_logger
from ..utils.exceptions import ExecutionError, ValidationError


class TaskCompletionStatus(Enum):
    """Task completion status levels"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    SUBSTANTIAL_PROGRESS = "substantial_progress"
    NEARLY_COMPLETE = "nearly_complete"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskProgress:
    """Task progress tracking"""
    task_description: str
    total_estimated_steps: int
    completed_steps: int = 0
    current_step_description: str = ""
    progress_percentage: float = 0.0
    last_update_time: float = field(default_factory=time.time)
    completion_indicators: List[str] = field(default_factory=list)
    missing_indicators: List[str] = field(default_factory=list)
    confidence_score: float = 0.0


@dataclass
class RobustnessConfig:
    """Configuration for robustness settings"""
    min_commands_per_task: int = 0  # No minimum limit
    max_commands_per_task: int = 0  # No maximum limit
    require_completion_validation: bool = True
    progress_check_interval: int = 3
    completion_confidence_threshold: float = 0.8
    allow_early_completion: bool = False
    force_full_execution: bool = True


class TaskRobustnessManager:
    """
    Manages task execution robustness to ensure complete step execution
    """
    
    def __init__(self, config: Optional[RobustnessConfig] = None):
        self.config = config or RobustnessConfig()
        self.logger = get_logger("task_robustness_manager")
        self.active_tasks: Dict[str, TaskProgress] = {}
        
        self.logger.info("Task robustness manager initialized", 
                       force_full_execution=self.config.force_full_execution,
                       min_commands=self.config.min_commands_per_task,
                       max_commands=self.config.max_commands_per_task)
    
    def start_task_execution(self, task_description: str, estimated_steps: int = 5) -> str:
        """Initialize tracking for a new task execution"""
        task_id = f"task_{int(time.time() * 1000)}"
        
        progress = TaskProgress(
            task_description=task_description,
            total_estimated_steps=estimated_steps,
            completed_steps=0,
            progress_percentage=0.0,
            confidence_score=0.0
        )
        
        self.active_tasks[task_id] = progress
        
        self.logger.info("Started tracking task execution",
                        task_id=task_id,
                        task_description=task_description,
                        estimated_steps=estimated_steps)
        
        return task_id
    
    def update_task_progress(self, task_id: str, command_description: str, 
                           completion_indicators: List[str] = None,
                           missing_indicators: List[str] = None) -> TaskProgress:
        """Update progress for an active task"""
        if task_id not in self.active_tasks:
            raise ValidationError(f"Task ID not found: {task_id}")
        
        progress = self.active_tasks[task_id]
        progress.completed_steps += 1
        progress.current_step_description = command_description
        progress.last_update_time = time.time()
        
        if completion_indicators:
            progress.completion_indicators.extend(completion_indicators)
        
        if missing_indicators:
            progress.missing_indicators.extend(missing_indicators)
        
        # Calculate progress percentage
        progress.progress_percentage = min(100.0, (progress.completed_steps / progress.total_estimated_steps) * 100)
        
        # Calculate confidence score based on multiple factors
        progress.confidence_score = self._calculate_confidence_score(progress)
        
        self.logger.debug("Updated task progress",
                         task_id=task_id,
                         completed_steps=progress.completed_steps,
                         progress_percentage=progress.progress_percentage,
                         confidence_score=progress.confidence_score)
        
        return progress
    
    def should_allow_task_completion(self, task_id: str, command_text: str) -> Tuple[bool, str]:
        """
        Determine if a task should be allowed to complete
        Returns (allow_completion, reason)
        """
        if task_id not in self.active_tasks:
            self.logger.warning("Task ID not found for completion check", task_id=task_id)
            return True, "Task not found in tracking"
        
        progress = self.active_tasks[task_id]
        
        # Force full execution mode - don't allow early completion
        if self.config.force_full_execution:
            # Check minimum command requirement (disabled when min_commands_per_task is 0)
            if self.config.min_commands_per_task > 0 and progress.completed_steps < self.config.min_commands_per_task:
                reason = f"Insufficient commands executed ({progress.completed_steps}/{self.config.min_commands_per_task} minimum)"
                self.logger.info("Blocking early completion - minimum commands not met",
                               task_id=task_id, reason=reason)
                return False, reason
            
            # Check if task appears genuinely complete
            if not self._is_task_genuinely_complete(progress, command_text):
                reason = "Task does not appear genuinely complete based on progress analysis"
                self.logger.info("Blocking completion - task not genuinely complete",
                               task_id=task_id, reason=reason)
                return False, reason
            
            # Check confidence threshold
            if progress.confidence_score < self.config.completion_confidence_threshold:
                reason = f"Confidence score too low ({progress.confidence_score:.2f} < {self.config.completion_confidence_threshold})"
                self.logger.info("Blocking completion - confidence too low",
                               task_id=task_id, reason=reason)
                return False, reason
        
        # If we reach here, completion is allowed
        reason = "Task completion approved"
        self.logger.info("Allowing task completion",
                        task_id=task_id, reason=reason,
                        completed_steps=progress.completed_steps,
                        confidence_score=progress.confidence_score)
        
        return True, reason
    
    def _is_task_genuinely_complete(self, progress: TaskProgress, command_text: str) -> bool:
        """Analyze if a task is genuinely complete based on multiple indicators"""
        
        # Check if this is an END command
        is_end_command = command_text.strip().upper() == "END"
        
        if not is_end_command:
            return False
        
        # Analyze completion indicators
        completion_score = 0
        
        # Progress percentage check
        if progress.progress_percentage >= 80.0:
            completion_score += 2
        elif progress.progress_percentage >= 60.0:
            completion_score += 1
        
        # Step count check
        if progress.completed_steps >= progress.total_estimated_steps:
            completion_score += 2
        elif progress.completed_steps >= progress.total_estimated_steps * 0.8:
            completion_score += 1
        
        # Completion indicators check
        if len(progress.completion_indicators) >= 3:
            completion_score += 2
        elif len(progress.completion_indicators) >= 1:
            completion_score += 1
        
        # Missing indicators check (negative factor)
        if len(progress.missing_indicators) == 0:
            completion_score += 2
        elif len(progress.missing_indicators) <= 2:
            completion_score += 1
        
        # Require at least 4 points for genuine completion
        is_complete = completion_score >= 4
        
        self.logger.debug("Task genuine completion analysis",
                         task_progress=progress.progress_percentage,
                         completed_steps=progress.completed_steps,
                         completion_indicators=len(progress.completion_indicators),
                         missing_indicators=len(progress.missing_indicators),
                         completion_score=completion_score,
                         is_complete=is_complete)
        
        return is_complete
    
    def _calculate_confidence_score(self, progress: TaskProgress) -> float:
        """Calculate confidence score for task completion"""
        score = 0.0
        
        # Base score from progress percentage
        score += (progress.progress_percentage / 100.0) * 0.4
        
        # Step completion factor
        step_ratio = min(1.0, progress.completed_steps / max(1, progress.total_estimated_steps))
        score += step_ratio * 0.3
        
        # Completion indicators factor
        indicator_score = min(1.0, len(progress.completion_indicators) / 5.0)
        score += indicator_score * 0.2
        
        # Missing indicators penalty
        missing_penalty = min(0.2, len(progress.missing_indicators) * 0.05)
        score -= missing_penalty
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, score))
    
    def get_task_status(self, task_id: str) -> TaskCompletionStatus:
        """Get current status of a task"""
        if task_id not in self.active_tasks:
            return TaskCompletionStatus.NOT_STARTED
        
        progress = self.active_tasks[task_id]
        
        if progress.confidence_score >= 0.9:
            return TaskCompletionStatus.COMPLETED
        elif progress.confidence_score >= 0.7:
            return TaskCompletionStatus.NEARLY_COMPLETE
        elif progress.confidence_score >= 0.5:
            return TaskCompletionStatus.SUBSTANTIAL_PROGRESS
        elif progress.completed_steps > 0:
            return TaskCompletionStatus.IN_PROGRESS
        else:
            return TaskCompletionStatus.NOT_STARTED
    
    def should_continue_task_execution(self, task_id: str, command_count: int) -> Tuple[bool, str]:
        """Determine if task execution should continue"""
        if task_id not in self.active_tasks:
            return False, "Task not found in tracking"
        
        progress = self.active_tasks[task_id]
        
        # Check maximum command limit (disabled when max_commands_per_task is 0)
        if self.config.max_commands_per_task > 0 and command_count >= self.config.max_commands_per_task:
            reason = f"Maximum command limit reached ({command_count}/{self.config.max_commands_per_task})"
            self.logger.warning("Stopping task execution - max commands reached",
                              task_id=task_id, reason=reason)
            return False, reason
        
        # Check if task is complete
        status = self.get_task_status(task_id)
        if status == TaskCompletionStatus.COMPLETED:
            reason = "Task marked as completed"
            self.logger.info("Task execution complete", task_id=task_id, reason=reason)
            return False, reason
        
        # Continue execution
        return True, "Continue execution"
    
    def end_task_execution(self, task_id: str, final_status: TaskCompletionStatus) -> Dict[str, Any]:
        """End task execution and return summary"""
        if task_id not in self.active_tasks:
            raise ValidationError(f"Task ID not found: {task_id}")
        
        progress = self.active_tasks[task_id]
        
        summary = {
            "task_id": task_id,
            "task_description": progress.task_description,
            "final_status": final_status.value,
            "total_steps": progress.total_estimated_steps,
            "completed_steps": progress.completed_steps,
            "progress_percentage": progress.progress_percentage,
            "confidence_score": progress.confidence_score,
            "completion_indicators": progress.completion_indicators,
            "missing_indicators": progress.missing_indicators,
            "execution_time": time.time() - (progress.last_update_time - 300)  # Approximate
        }
        
        # Remove from active tasks
        del self.active_tasks[task_id]
        
        self.logger.info("Task execution ended", **summary)
        
        return summary
    
    def get_active_task_summary(self) -> Dict[str, Any]:
        """Get summary of all active tasks"""
        return {
            "total_active_tasks": len(self.active_tasks),
            "tasks": {
                task_id: {
                    "description": progress.task_description,
                    "progress": progress.progress_percentage,
                    "completed_steps": progress.completed_steps,
                    "confidence": progress.confidence_score,
                    "status": self.get_task_status(task_id).value
                }
                for task_id, progress in self.active_tasks.items()
            }
        }


# Global instance
_global_robustness_manager: Optional[TaskRobustnessManager] = None


def get_task_robustness_manager(config: Optional[RobustnessConfig] = None) -> TaskRobustnessManager:
    """Get global task robustness manager instance"""
    global _global_robustness_manager
    
    if _global_robustness_manager is None:
        _global_robustness_manager = TaskRobustnessManager(config)
    
    return _global_robustness_manager
