"""
Model Runner for AI Agent System
2-Phase Vision-Only Architecture: Ollama Cloud Models Only
"""

import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from .vision_api_client import VisionAPIClient, APIRequest, APIResponse, APIProvider
from ..utils.exceptions import APIError, ValidationError, TaskGenerationError
from ..utils.logger import get_logger
from ..utils.config import load_config


class TaskType(Enum):
    """Task types for 2-Phase Architecture"""
    TASK_GENERATION = "task_generation"
    COMMAND_PARSING = "command_parsing"
    REBUILD_PLANNING = "rebuild_planning"


@dataclass
class ModelRequest:
    """Model request structure"""
    task_type: TaskType
    prompt: str
    image_data: Optional[bytes] = None
    image_format: str = "PNG"
    context: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 30


@dataclass
class ModelResponse:
    """Model response structure"""
    success: bool
    content: str
    task_type: TaskType
    model: str
    provider: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    latency: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PromptTemplate:
    """Prompt template manager"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates for 2-Phase Architecture"""
        return {
            TaskType.TASK_GENERATION.value: """You are an AI assistant operating the OS. Analyze the user's instructions and the current screen state to generate a specific GUI automation task list.

User Instruction: {instruction}

Current Screen: [Screenshot provided]

Analyze the screen and generate a numbered list of specific tasks to achieve the user's instruction. Each task must meet the following conditions:
1. Be specific and actionable
2. Be listed in sequential order
3. Focus on a single action
4. Be feasible given the current screen state

Response format:
1. [First specific task]
2. [Second specific task]
3. [Third specific task]
...

Provide only the numbered list; do not include additional text.""",
            
            TaskType.COMMAND_PARSING.value: """You are an AI assistant operating the OS. Convert the task description into specific GUI automation commands.

Task Description: {task_description}

Current Screen: [Screenshot provided]
Previous Screen: [previous screenshot provided]
Previous Command: {previous_command}

Available Commands:
- click(x, y) - Click at normalized coordinates (0.0-1.0)
- double_click(x, y) - Double-click at normalized coordinates
- right_click(x, y) - Right-click at normalized coordinates
- text("content") - Input text content
- key(keys) - Press key combination (e.g., "ctrl+c", "enter", "cmd+space")
- drag(start_x, start_y, end_x, end_y) - Drag from start to end coordinates
- scroll(direction, amount) - Scroll (direction: up/down/left/right, amount: 1-10)
- END - End task execution
- REBUILD - Trigger self-repair process when stuck or failed

Use normalized coordinates (0.0-1.0). The top-left corner of the screen is (0.0, 0.0), the bottom-right corner is (1.0, 1.0).

Convert the task description into the appropriate command. Assess the situation and output only one optimal command to execute next.

Important: Do not output descriptions; output only commands. Write only one command per line.
If you detect procedure inconsistencies, loops, or stuck situations, immediately output REBUILD to trigger self-repair.

Example:
click(0.5, 0.3)
text("Search query")
key("enter")
END""",
            
            TaskType.REBUILD_PLANNING.value: """You are an AI recovery specialist. Analyze the failure context and generate a revised task list to achieve the original goal.

Original Instruction: {original_instruction}
Original Task List: {original_task_list}
Execution History: {execution_history}
Current Screen: [Screenshot provided]

Analyze the above data and identify what went wrong. Create a revised numbered task list that addresses the failure and provides a clear path to achieve the original goal.

Requirements:
1. Learn from the execution history to avoid repeating mistakes
2. Adapt to the current screen state
3. Provide specific, actionable tasks
4. Maintain the original goal focus
5. Break down complex steps if needed

Response format:
1. [Revised first task]
2. [Revised second task]
3. [Revised third task]
...

Provide only the numbered list; do not include additional text.""",
        }
    
    def get_template(self, task_type: TaskType) -> str:
        """Get prompt template for task type"""
        return self.templates.get(task_type.value, self.templates[TaskType.TASK_GENERATION.value])
    
    def format_prompt(self, task_type: TaskType, **kwargs) -> str:
        """Format prompt template with variables"""
        template = self.get_template(task_type)
        return template.format(**kwargs)


class ModelRunner:
    """2-Phase Architecture Model Runner: Ollama Cloud Models Only"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config().api.__dict__
        self.logger = get_logger("model_runner")
        
        # Initialize components
        self.vision_client = VisionAPIClient(self.config)
        self.prompt_template = PromptTemplate()
        
        self.logger.info(
            "Model runner initialized for 2-Phase Architecture",
            provider="ollama",
            model=self.config.get("local_model", "ollama-cloud-model:latest"),
        )
    
    def run_model(self, request: ModelRequest) -> ModelResponse:
        """Run AI model for 2-Phase Architecture"""
        start_time = time.time()
        
        try:
            # Validate request
            self._validate_request(request)
            
            # Format prompt
            prompt = self._format_prompt(request)
            
            # Create API request for AI model via Ollama
            api_request = APIRequest(
                prompt=prompt,
                image_data=request.image_data,
                image_format=request.image_format,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                model=self.config.get("local_model", "gemini-3-flash-preview:latest"),
                provider=APIProvider.OLLAMA,
            )
            
            # Make API call
            api_response = self.vision_client.analyze_image(api_request)
            
            # Create model response
            model_response = ModelResponse(
                success=api_response.success,
                content=api_response.content,
                task_type=request.task_type,
                model=api_response.model,
                provider=api_response.provider,
                tokens_used=api_response.tokens_used,
                cost=api_response.cost,
                latency=time.time() - start_time,
                error=api_response.error,
            )
            
            # Log success
            self.logger.info(
                "AI model execution successful",
                task_type=request.task_type.value,
                model=api_response.model,
                provider=api_response.provider,
                tokens_used=api_response.tokens_used,
                latency=model_response.latency,
            )
            
            return model_response
            
        except ValidationError as e:
            # Re-raise validation errors - these should not be masked
            raise
        except Exception as e:
            # Create error response for other exceptions
            error_response = ModelResponse(
                success=False,
                content="",
                task_type=request.task_type,
                model="",
                provider="",
                latency=time.time() - start_time,
                error=str(e),
            )
            
            self.logger.error(
                "AI model execution failed",
                task_type=request.task_type.value,
                error=str(e),
                latency=error_response.latency,
            )
            
            return error_response
    
    def _validate_request(self, request: ModelRequest):
        """Validate model request"""
        if not request.prompt:
            raise ValidationError("Prompt cannot be empty", "prompt", request.prompt)
        
        if request.max_tokens < 1 or request.max_tokens > 4000:
            raise ValidationError("Invalid max_tokens", "max_tokens", request.max_tokens)
        
        if not (0.0 <= request.temperature <= 2.0):
            raise ValidationError("Invalid temperature", "temperature", request.temperature)
        
        if request.task_type not in TaskType:
            raise ValidationError("Invalid task type", "task_type", request.task_type)
    
    def _format_prompt(self, request: ModelRequest) -> str:
        """Format prompt based on task type and context"""
        # Get template
        template = self.prompt_template.get_template(request.task_type)
        
        # Prepare formatting variables
        format_vars = {
            "instruction": request.prompt,
            "task_description": request.prompt,
        }
        
        # Add context variables
        if request.context:
            format_vars.update(request.context)
            # Handle previous screenshot and command for command parsing
            if request.task_type == TaskType.COMMAND_PARSING:
                if "previous_screenshot" in request.context:
                    format_vars["previous_screenshot"] = "[Previous screenshot provided]"
                if "previous_command" in request.context:
                    format_vars["previous_command"] = request.context["previous_command"]
                else:
                    format_vars["previous_command"] = "None"
        
        # Format template
        try:
            formatted_prompt = template.format(**format_vars)
        except KeyError as e:
            # Missing template variable - use basic prompt
            self.logger.warning(f"Template variable missing: {e}")
            formatted_prompt = request.prompt
        except Exception as e:
            # Other formatting errors - use basic prompt
            self.logger.error(f"Template formatting error: {e}")
            formatted_prompt = request.prompt
        
        return formatted_prompt
    
    # Task-specific methods for 2-Phase Architecture
    
    def generate_tasks(self, instruction: str, screenshot: bytes, context: Optional[Dict[str, Any]] = None) -> ModelResponse:
        """Phase 1: Generate task list from instruction and screenshot"""
        request = ModelRequest(
            task_type=TaskType.TASK_GENERATION,
            prompt=instruction,
            image_data=screenshot,
            context=context or {},
            parameters={},
        )
        
        return self.run_model(request)
    
    def parse_command(self, task_description: str, screenshot: bytes, context: Optional[Dict[str, Any]] = None, previous_screenshot: Optional[bytes] = None, previous_command: Optional[str] = None) -> ModelResponse:
        """Phase 2: Parse task description into automation command"""
        # Enhanced context with previous screenshot and command
        enhanced_context = context or {}
        if previous_screenshot:
            enhanced_context["previous_screenshot"] = previous_screenshot
        if previous_command:
            enhanced_context["previous_command"] = previous_command
        
        request = ModelRequest(
            task_type=TaskType.COMMAND_PARSING,
            prompt=task_description,
            image_data=screenshot,
            context=enhanced_context,
            parameters={},
        )
        
        return self.run_model(request)
    
    def generate_rebuild_plan(self, original_instruction: str, original_task_list: List[str], execution_history: List[str], current_screenshot: bytes) -> ModelResponse:
        """Generate revised task list for recovery/rebuild scenario"""
        # Validate inputs
        if not original_instruction:
            raise ValidationError("Original instruction cannot be empty", "original_instruction", original_instruction)
        
        if original_task_list is None:
            original_task_list = []
        
        if execution_history is None:
            execution_history = []
        
        if not current_screenshot:
            raise ValidationError("Current screenshot cannot be empty", "current_screenshot", current_screenshot)
        
        # Format context variables
        context = {
            "original_instruction": original_instruction,
            "original_task_list": "\n".join([f"{i+1}. {task}" for i, task in enumerate(original_task_list)]) if original_task_list else "No previous tasks",
            "execution_history": "\n".join([f"- {cmd}" for cmd in execution_history[-10:]]) if execution_history else "No execution history"
        }
        
        request = ModelRequest(
            task_type=TaskType.REBUILD_PLANNING,
            prompt=original_instruction,  # Primary prompt is the original instruction
            image_data=current_screenshot,
            context=context,
            parameters={},
        )
        
        return self.run_model(request)


# Global model runner instance
_model_runner: Optional[ModelRunner] = None


def get_model_runner() -> ModelRunner:
    """Get global model runner instance"""
    global _model_runner
    
    if _model_runner is None:
        _model_runner = ModelRunner()
    
    return _model_runner
