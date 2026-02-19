"""
Task Completion Verifier Module
Implements verification system using Gemini 3 Flash to validate task completion
before the AI agent terminates.
"""

import json
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

from ..external_integration.model_runner import ModelRunner, TaskType, ModelResponse
from ..external_integration.vision_api_client import VisionAPIClient, APIRequest, APIResponse, APIProvider
from ..platform_abstraction.screenshot_capture import ScreenshotCapture
from ..utils.exceptions import VerificationError, APIError
from ..utils.logger import get_logger
from .save_command import SaveCommand, get_save_command


class VerificationResult(Enum):
    """Verification result types"""
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    UNCERTAIN = "uncertain"
    ERROR = "error"


@dataclass
class TaskVerification:
    """Task verification result structure"""
    result: VerificationResult
    confidence: float  # 0.0 to 1.0
    reasoning: str
    missing_steps: List[str]
    suggestions: List[str]
    should_regenerate: bool
    metadata: Dict[str, Any]


class TaskCompletionVerifier:
    """
    Task Completion Verifier using Gemini 3 Flash
    
    This module verifies whether the AI agent has truly completed the user's task
    by analyzing the current screen state, accumulated save command logs, and the original instruction.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("task_completion_verifier")
        
        # Load configuration using the config system
        from ..utils.config import load_config
        full_config = load_config()
        
        # Initialize components
        self.model_runner = ModelRunner(self.config)
        self.screenshot_capture = ScreenshotCapture()
        self.save_command = get_save_command()
        
        # Verification settings from config
        self.confidence_threshold = getattr(full_config.verification, 'confidence_threshold', 0.8)
        self.max_verification_attempts = getattr(full_config.verification, 'max_verification_attempts', 3)
        self.verification_enabled = getattr(full_config.verification, 'enabled', True)
        
        self.logger.info("Task completion verifier initialized with Gemini 3 Flash")
    
    def verify_task_completion(self, original_instruction: str, session_id: Optional[str] = None) -> TaskVerification:
        """
        Verify if the task has been completed successfully
        
        Args:
            original_instruction: The user's original task instruction
            session_id: Optional session ID for save command context
            
        Returns:
            TaskVerification: Detailed verification result
        """
        # Check if verification is enabled
        if not self.verification_enabled:
            self.logger.info("Task verification is disabled, returning completed by default")
            return TaskVerification(
                result=VerificationResult.COMPLETED,
                confidence=1.0,
                reasoning="Verification disabled - assuming completion",
                missing_steps=[],
                suggestions=[],
                should_regenerate=False,
                metadata={"verification_disabled": True}
            )
        
        self.logger.info(f"Starting task completion verification for: {original_instruction}")
        
        try:
            # Capture current screen state
            screenshot_data, screenshot_metadata = self.screenshot_capture.capture_screenshot()
            
            # Get save command logs
            save_logs = self._get_save_command_logs(session_id)
            
            # Create verification prompt
            verification_prompt = self._create_verification_prompt(
                original_instruction, save_logs, screenshot_metadata
            )
            
            # Call Gemini 3 Flash for verification
            verification_response = self._call_gemini_for_verification(
                verification_prompt, screenshot_data
            )
            
            # Parse verification response
            verification_result = self._parse_verification_response(verification_response)
            
            self.logger.info(
                f"Task verification completed",
                result=verification_result.result.value,
                confidence=verification_result.confidence,
                should_regenerate=verification_result.should_regenerate
            )
            
            return verification_result
            
        except Exception as e:
            self.logger.error(f"Task verification failed: {e}")
            return TaskVerification(
                result=VerificationResult.ERROR,
                confidence=0.0,
                reasoning=f"Verification error: {str(e)}",
                missing_steps=[],
                suggestions=[],
                should_regenerate=False,
                metadata={"error": str(e)}
            )
    
    def _get_save_command_logs(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Extract and format save command logs for verification"""
        try:
            # Get recent save entries
            recent_entries = self.save_command.get_recent_saves(count=20)  # Get last 20 entries
            
            # Format logs for analysis
            logs = {
                "total_entries": len(recent_entries),
                "has_failures": self.save_command.has_failures(),
                "extracted_info": self.save_command.get_extracted_information(),
                "failure_coordinates": self.save_command.get_failure_coordinates(),
                "entries": []
            }
            
            for entry in recent_entries:
                entry_data = {
                    "timestamp": entry.timestamp,
                    "content": entry.content,
                    "content_type": entry.content_type.value,
                    "operation_command": entry.operation_command,
                    "coordinates": entry.coordinates,
                    "visual_feedback": entry.visual_feedback,
                    "extracted_info": entry.extracted_info,
                    "failure_details": entry.failure_details
                }
                logs["entries"].append(entry_data)
            
            return logs
            
        except Exception as e:
            self.logger.error(f"Failed to get save command logs: {e}")
            return {
                "total_entries": 0,
                "has_failures": False,
                "extracted_info": {},
                "failure_coordinates": [],
                "entries": [],
                "error": str(e)
            }
    
    def _create_verification_prompt(self, original_instruction: str, save_logs: Dict[str, Any], screenshot_metadata: Any) -> str:
        """Create verification prompt for Gemini 3 Flash"""
        
        # Format save logs for prompt
        formatted_logs = self._format_logs_for_prompt(save_logs)
        
        prompt = f"""You are a task completion verification AI. Analyze whether the user's task has been completed successfully.

USER'S ORIGINAL INSTRUCTION:
{original_instruction}

CURRENT SCREEN STATE:
[Screenshot provided - analyze the visual state]

EXECUTION LOGS:
{formatted_logs}

SCREENSHOT METADATA:
{json.dumps(screenshot_metadata.__dict__ if hasattr(screenshot_metadata, '__dict__') else {}, indent=2)}

ANALYSIS REQUIREMENTS:
1. Examine the current screen state to see if the task goal appears to be achieved
2. Review the execution logs to understand what actions were taken
3. Identify any missing steps or incomplete actions
4. Assess confidence in completion (0.0 to 1.0)

RESPONSE FORMAT (JSON):
{{
    "result": "completed|incomplete|uncertain|error",
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of your analysis",
    "missing_steps": ["List of specific missing actions if incomplete"],
    "suggestions": ["List of suggestions to complete the task if incomplete"],
    "should_regenerate": true/false
}}

ANALYSIS CRITERIA:
- "completed": Task goal is clearly achieved with high confidence (>0.8)
- "incomplete": Task goal is clearly not achieved or significant steps missing
- "uncertain": Cannot determine completion with reasonable confidence
- "error": Analysis failed due to technical issues

Focus on the actual outcome visible on screen versus the intended goal of the instruction."""
        
        return prompt
    
    def _format_logs_for_prompt(self, save_logs: Dict[str, Any]) -> str:
        """Format save logs for inclusion in verification prompt"""
        try:
            lines = []
            lines.append(f"Total log entries: {save_logs.get('total_entries', 0)}")
            lines.append(f"Has failures: {save_logs.get('has_failures', False)}")
            lines.append(f"Extracted information: {json.dumps(save_logs.get('extracted_info', {}), indent=2)}")
            
            if save_logs.get('failure_coordinates'):
                lines.append(f"Failed coordinates: {save_logs['failure_coordinates']}")
            
            lines.append("\nRecent execution entries:")
            for i, entry in enumerate(save_logs.get('entries', [])[:10]):  # Limit to 10 most recent
                lines.append(f"{i+1}. [{entry.get('content_type', 'unknown')}] {entry.get('content', 'No content')}")
                if entry.get('operation_command'):
                    lines.append(f"   Command: {entry['operation_command']}")
                if entry.get('visual_feedback'):
                    lines.append(f"   Feedback: {entry['visual_feedback']}")
                if entry.get('failure_details'):
                    lines.append(f"   Failure: {entry['failure_details']}")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error formatting logs: {str(e)}"
    
    def _call_gemini_for_verification(self, prompt: str, screenshot_data: bytes) -> ModelResponse:
        """Call Gemini 3 Flash for task verification"""
        try:
            # Create API request for Gemini 3 Flash
            api_request = APIRequest(
                prompt=prompt,
                image_data=screenshot_data,
                image_format="PNG",
                max_tokens=5000,
                temperature=0.3,  # Lower temperature for more consistent verification
                model="gemini-3-flash-preview:latest",
                provider=APIProvider.OLLAMA,
            )
            
            # Make API call through vision client
            vision_client = VisionAPIClient(self.config)
            api_response = vision_client.analyze_image(api_request)
            
            # Convert to ModelResponse
            return ModelResponse(
                success=api_response.success,
                content=api_response.content,
                task_type=TaskType.COMMAND_PARSING,  # Reuse existing task type
                model=api_response.model,
                provider=api_response.provider,
                tokens_used=api_response.tokens_used,
                cost=api_response.cost,
                latency=api_response.latency,
                error=api_response.error,
            )
            
        except Exception as e:
            self.logger.error(f"Failed to call Gemini for verification: {e}")
            return ModelResponse(
                success=False,
                content="",
                task_type=TaskType.COMMAND_PARSING,
                model="gemini-3-flash-preview:latest",
                provider="ollama",
                error=str(e),
            )
    
    def _parse_verification_response(self, response: ModelResponse) -> TaskVerification:
        """Parse Gemini's response into TaskVerification object"""
        try:
            if not response.success or not response.content:
                return TaskVerification(
                    result=VerificationResult.ERROR,
                    confidence=0.0,
                    reasoning="No response from verification model",
                    missing_steps=[],
                    suggestions=[],
                    should_regenerate=False,
                    metadata={"error": response.error or "Empty response"}
                )
            
            # Try to parse JSON response
            try:
                verification_data = json.loads(response.content.strip())
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    verification_data = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON from response")
            
            # Validate and map result
            result_str = verification_data.get("result", "uncertain").lower()
            result_map = {
                "completed": VerificationResult.COMPLETED,
                "incomplete": VerificationResult.INCOMPLETE,
                "uncertain": VerificationResult.UNCERTAIN,
                "error": VerificationResult.ERROR
            }
            result = result_map.get(result_str, VerificationResult.UNCERTAIN)
            
            # Determine if regeneration should be recommended
            should_regenerate = verification_data.get("should_regenerate", False)
            if not should_regenerate:
                # Auto-determine based on result and confidence
                confidence = verification_data.get("confidence", 0.0)
                should_regenerate = (
                    result == VerificationResult.INCOMPLETE or
                    (result == VerificationResult.UNCERTAIN and confidence < 0.6)
                )
            
            return TaskVerification(
                result=result,
                confidence=float(verification_data.get("confidence", 0.0)),
                reasoning=verification_data.get("reasoning", "No reasoning provided"),
                missing_steps=verification_data.get("missing_steps", []),
                suggestions=verification_data.get("suggestions", []),
                should_regenerate=should_regenerate,
                metadata={
                    "raw_response": response.content,
                    "model": response.model,
                    "tokens_used": response.tokens_used,
                    "latency": response.latency
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse verification response: {e}")
            return TaskVerification(
                result=VerificationResult.ERROR,
                confidence=0.0,
                reasoning=f"Failed to parse verification response: {str(e)}",
                missing_steps=[],
                suggestions=[],
                should_regenerate=False,
                metadata={"parse_error": str(e), "raw_response": response.content}
            )


# Global verifier instance
_global_verifier: Optional[TaskCompletionVerifier] = None


def get_task_completion_verifier(config: Optional[Dict[str, Any]] = None) -> TaskCompletionVerifier:
    """Get global task completion verifier instance"""
    global _global_verifier
    
    if _global_verifier is None:
        _global_verifier = TaskCompletionVerifier(config)
    
    return _global_verifier


def verify_task_completion(original_instruction: str, session_id: Optional[str] = None) -> TaskVerification:
    """
    Global function to verify task completion
    
    Args:
        original_instruction: The user's original task instruction
        session_id: Optional session ID for save command context
        
    Returns:
        TaskVerification: Detailed verification result
    """
    verifier = get_task_completion_verifier()
    return verifier.verify_task_completion(original_instruction, session_id)
