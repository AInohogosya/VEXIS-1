"""
Enhanced Task Verifier for Robust Task Completion
Integrates with task robustness manager to provide comprehensive verification
"""

import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from .task_completion_verifier import TaskVerification, VerificationResult
from .task_robustness_manager import TaskCompletionStatus, get_task_robustness_manager
from ..utils.logger import get_logger


@dataclass
class EnhancedVerificationResult:
    """Enhanced verification result combining multiple verification methods"""
    original_verification: TaskVerification
    robustness_status: TaskCompletionStatus
    robustness_summary: Dict[str, Any]
    final_decision: VerificationResult
    combined_confidence: float
    detailed_reasoning: str
    should_continue_execution: bool
    additional_steps_needed: List[str]


class EnhancedTaskVerifier:
    """
    Enhanced task verifier that combines traditional verification with robustness analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("enhanced_task_verifier")
        self.robustness_manager = get_task_robustness_manager()
        
        self.logger.info("Enhanced task verifier initialized")
    
    def verify_task_completion_enhanced(self, 
                                      original_instruction: str,
                                      session_id: Optional[str] = None,
                                      task_robustness_summary: Optional[Dict[str, Any]] = None) -> EnhancedVerificationResult:
        """
        Perform enhanced task verification combining traditional and robustness-based verification
        
        Args:
            original_instruction: The user's original task instruction
            session_id: Optional session ID for save command context
            task_robustness_summary: Optional robustness summary from task execution
            
        Returns:
            EnhancedVerificationResult: Comprehensive verification result
        """
        self.logger.info("Starting enhanced task verification",
                        instruction=original_instruction,
                        has_robustness_summary=task_robustness_summary is not None)
        
        try:
            # Get traditional verification
            from .task_completion_verifier import get_task_completion_verifier
            traditional_verifier = get_task_completion_verifier(self.config)
            traditional_verification = traditional_verifier.verify_task_completion(
                original_instruction, session_id
            )
            
            # Get robustness-based verification
            robustness_verification = self._perform_robustness_verification(
                original_instruction, task_robustness_summary
            )
            
            # Combine verification results
            enhanced_result = self._combine_verification_results(
                traditional_verification, robustness_verification, task_robustness_summary
            )
            
            self.logger.info("Enhanced verification completed",
                            final_decision=enhanced_result.final_decision.value,
                            combined_confidence=enhanced_result.combined_confidence,
                            should_continue=enhanced_result.should_continue_execution)
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Enhanced verification failed: {e}")
            # Return fallback result
            return EnhancedVerificationResult(
                original_verification=TaskVerification(
                    result=VerificationResult.ERROR,
                    confidence=0.0,
                    reasoning=f"Enhanced verification error: {str(e)}",
                    missing_steps=[],
                    suggestions=[],
                    should_regenerate=False,
                    metadata={"error": str(e)}
                ),
                robustness_status=TaskCompletionStatus.FAILED,
                robustness_summary={},
                final_decision=VerificationResult.ERROR,
                combined_confidence=0.0,
                detailed_reasoning=f"Enhanced verification failed: {str(e)}",
                should_continue_execution=False,
                additional_steps_needed=[]
            )
    
    def _perform_robustness_verification(self, 
                                       original_instruction: str,
                                       task_robustness_summary: Optional[Dict[str, Any]]) -> Tuple[VerificationResult, float, str]:
        """Perform robustness-based verification"""
        if not task_robustness_summary:
            return VerificationResult.UNCERTAIN, 0.5, "No robustness data available"
        
        confidence_score = task_robustness_summary.get("confidence_score", 0.0)
        progress_percentage = task_robustness_summary.get("progress_percentage", 0.0)
        completed_steps = task_robustness_summary.get("completed_steps", 0)
        total_steps = task_robustness_summary.get("total_steps", 1)
        
        # Analyze robustness indicators
        completion_indicators = task_robustness_summary.get("completion_indicators", [])
        missing_indicators = task_robustness_summary.get("missing_indicators", [])
        
        # Determine robustness-based result
        if confidence_score >= 0.9 and progress_percentage >= 90.0:
            result = VerificationResult.COMPLETED
            reasoning = f"High confidence ({confidence_score:.2f}) and progress ({progress_percentage:.1f}%) indicate completion"
        elif confidence_score >= 0.7 and progress_percentage >= 70.0:
            result = VerificationResult.COMPLETED
            reasoning = f"Good confidence ({confidence_score:.2f}) and progress ({progress_percentage:.1f}%) indicate likely completion"
        elif confidence_score < 0.4 or progress_percentage < 40.0:
            result = VerificationResult.INCOMPLETE
            reasoning = f"Low confidence ({confidence_score:.2f}) or progress ({progress_percentage:.1f}%) indicates incomplete task"
        else:
            result = VerificationResult.UNCERTAIN
            reasoning = f"Moderate confidence ({confidence_score:.2f}) and progress ({progress_percentage:.1f}%) create uncertainty"
        
        # Adjust based on indicators
        if len(completion_indicators) >= 3 and len(missing_indicators) == 0:
            result = VerificationResult.COMPLETED
            reasoning += " - Strong completion indicators and no missing indicators"
        elif len(missing_indicators) > len(completion_indicators):
            if result == VerificationResult.COMPLETED:
                result = VerificationResult.UNCERTAIN
            reasoning += " - More missing indicators than completion indicators"
        
        self.logger.debug("Robustness verification analysis",
                         confidence_score=confidence_score,
                         progress_percentage=progress_percentage,
                         completion_indicators=len(completion_indicators),
                         missing_indicators=len(missing_indicators),
                         result=result.value)
        
        return result, confidence_score, reasoning
    
    def _combine_verification_results(self,
                                    traditional_verification: TaskVerification,
                                    robustness_verification: Tuple[VerificationResult, float, str],
                                    task_robustness_summary: Optional[Dict[str, Any]]) -> EnhancedVerificationResult:
        """Combine traditional and robustness verification results"""
        
        robustness_result, robustness_confidence, robustness_reasoning = robustness_verification
        
        # Get robustness status
        robustness_status = TaskCompletionStatus.IN_PROGRESS
        if task_robustness_summary:
            # Map from robustness summary to status
            confidence = task_robustness_summary.get("confidence_score", 0.0)
            if confidence >= 0.9:
                robustness_status = TaskCompletionStatus.COMPLETED
            elif confidence >= 0.7:
                robustness_status = TaskCompletionStatus.NEARLY_COMPLETE
            elif confidence >= 0.5:
                robustness_status = TaskCompletionStatus.SUBSTANTIAL_PROGRESS
            elif confidence > 0.0:
                robustness_status = TaskCompletionStatus.IN_PROGRESS
            else:
                robustness_status = TaskCompletionStatus.NOT_STARTED
        
        # Weight the different verification methods
        traditional_weight = 0.6
        robustness_weight = 0.4
        
        # Calculate combined confidence
        combined_confidence = (
            traditional_verification.confidence * traditional_weight +
            robustness_confidence * robustness_weight
        )
        
        # Determine final decision
        final_decision = self._determine_final_decision(
            traditional_verification.result,
            robustness_result,
            combined_confidence,
            traditional_verification.confidence,
            robustness_confidence
        )
        
        # Create detailed reasoning
        detailed_reasoning = self._create_detailed_reasoning(
            traditional_verification,
            robustness_result,
            robustness_reasoning,
            combined_confidence
        )
        
        # Determine if execution should continue
        should_continue_execution = self._should_continue_execution(
            final_decision, combined_confidence, robustness_status
        )
        
        # Identify additional steps needed
        additional_steps_needed = self._identify_additional_steps(
            traditional_verification, task_robustness_summary
        )
        
        return EnhancedVerificationResult(
            original_verification=traditional_verification,
            robustness_status=robustness_status,
            robustness_summary=task_robustness_summary or {},
            final_decision=final_decision,
            combined_confidence=combined_confidence,
            detailed_reasoning=detailed_reasoning,
            should_continue_execution=should_continue_execution,
            additional_steps_needed=additional_steps_needed
        )
    
    def _determine_final_decision(self,
                                traditional_result: VerificationResult,
                                robustness_result: VerificationResult,
                                combined_confidence: float,
                                traditional_confidence: float,
                                robustness_confidence: float) -> VerificationResult:
        """Determine final verification decision from multiple inputs"""
        
        # If both methods agree, use that result
        if traditional_result == robustness_result:
            return traditional_result
        
        # If one method strongly disagrees with the other
        confidence_diff = abs(traditional_confidence - robustness_confidence)
        
        if confidence_diff > 0.3:
            # Use the result from the more confident method
            if traditional_confidence > robustness_confidence:
                return traditional_result
            else:
                return robustness_result
        
        # If confidence levels are similar but results differ, be conservative
        if traditional_result == VerificationResult.COMPLETED and robustness_result == VerificationResult.INCOMPLETE:
            return VerificationResult.UNCERTAIN
        elif traditional_result == VerificationResult.INCOMPLETE and robustness_result == VerificationResult.COMPLETED:
            return VerificationResult.UNCERTAIN
        else:
            # Use the more conservative result
            result_priority = {
                VerificationResult.INCOMPLETE: 1,
                VerificationResult.UNCERTAIN: 2,
                VerificationResult.COMPLETED: 3,
                VerificationResult.ERROR: 0
            }
            
            return min([traditional_result, robustness_result], 
                      key=lambda x: result_priority.get(x, 0))
    
    def _create_detailed_reasoning(self,
                                traditional_verification: TaskVerification,
                                robustness_result: VerificationResult,
                                robustness_reasoning: str,
                                combined_confidence: float) -> str:
        """Create detailed reasoning combining all verification inputs"""
        
        reasoning_parts = []
        
        # Traditional verification reasoning
        reasoning_parts.append(f"Traditional verification: {traditional_verification.reasoning}")
        reasoning_parts.append(f"Traditional result: {traditional_verification.result.value} (confidence: {traditional_verification.confidence:.2f})")
        
        # Robustness verification reasoning
        reasoning_parts.append(f"Robustness verification: {robustness_reasoning}")
        reasoning_parts.append(f"Robustness result: {robustness_result.value}")
        
        # Combined analysis
        reasoning_parts.append(f"Combined confidence: {combined_confidence:.2f}")
        
        # Agreement analysis
        if traditional_verification.result == robustness_result:
            reasoning_parts.append("Both verification methods agree on the result")
        else:
            reasoning_parts.append("Verification methods disagree - using conservative approach")
        
        return "\n".join(reasoning_parts)
    
    def _should_continue_execution(self,
                                 final_decision: VerificationResult,
                                 combined_confidence: float,
                                 robustness_status: TaskCompletionStatus) -> bool:
        """Determine if task execution should continue"""
        
        # Don't continue if there's an error
        if final_decision == VerificationResult.ERROR:
            return False
        
        # Continue if task is incomplete
        if final_decision == VerificationResult.INCOMPLETE:
            return True
        
        # Continue if uncertain and confidence is low
        if final_decision == VerificationResult.UNCERTAIN and combined_confidence < 0.6:
            return True
        
        # Continue if robustness status indicates incomplete progress
        if robustness_status in [TaskCompletionStatus.IN_PROGRESS, TaskCompletionStatus.SUBSTANTIAL_PROGRESS]:
            return True
        
        # Otherwise, don't continue
        return False
    
    def _identify_additional_steps(self,
                                 traditional_verification: TaskVerification,
                                 task_robustness_summary: Optional[Dict[str, Any]]) -> List[str]:
        """Identify additional steps needed for task completion"""
        
        additional_steps = []
        
        # Add steps from traditional verification
        additional_steps.extend(traditional_verification.missing_steps)
        
        # Add steps from robustness analysis
        if task_robustness_summary:
            missing_indicators = task_robustness_summary.get("missing_indicators", [])
            additional_steps.extend(missing_indicators)
            
            # Add generic steps based on progress
            progress_percentage = task_robustness_summary.get("progress_percentage", 0.0)
            if progress_percentage < 50.0:
                additional_steps.append("Continue executing core task steps")
            elif progress_percentage < 80.0:
                additional_steps.append("Complete remaining task verification steps")
        
        # Remove duplicates and limit to reasonable number
        unique_steps = list(set(additional_steps))[:5]
        
        return unique_steps


# Global instance
_global_enhanced_verifier: Optional[EnhancedTaskVerifier] = None


def get_enhanced_task_verifier(config: Optional[Dict[str, Any]] = None) -> EnhancedTaskVerifier:
    """Get global enhanced task verifier instance"""
    global _global_enhanced_verifier
    
    if _global_enhanced_verifier is None:
        _global_enhanced_verifier = EnhancedTaskVerifier(config)
    
    return _global_enhanced_verifier
