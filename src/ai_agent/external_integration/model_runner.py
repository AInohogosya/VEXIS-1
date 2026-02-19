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


@dataclass
class ModelRequest:
    """Model request structure"""
    task_type: TaskType
    prompt: str
    image_data: Optional[bytes] = None
    image_format: str = "PNG"
    context: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None
    max_tokens: int = 5000
    temperature: float = 1.0
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

IMPORTANT: Please provide a MORE DETAILED step list. Break down complex actions into smaller, more granular steps. For example, instead of "Open the browser and search", break it into:
1. Click on the browser icon
2. Wait for browser to open
3. Click in the search bar
4. Type the search query
5. Press Enter

Each step should be atomic and executable as a single GUI action. Consider intermediate states, loading times, and necessary sub-steps that might be overlooked.

Response format:
1. [First detailed, atomic task]
2. [Second detailed, atomic task]
3. [Third detailed, atomic task]
...

Provide only the numbered list; do not include additional text.""",
            
            TaskType.COMMAND_PARSING.value: """You are an AI assistant operating the OS. Convert the task description into specific GUI automation commands.

Task Description: {task_description}

Current Screen: [Screenshot provided]
Previous Screen: [previous screenshot provided]
Previous Command: {previous_command}

YOUR PREVIOUS ACTIONS - COMPLETED STEPS BY YOU:
The following is a complete history of actions that YOU have already performed in this task. These are YOUR completed steps:
{previous_save_content}

CRITICAL UNDERSTANDING - THESE ARE YOUR ACTIONS:
- EVERY item in "YOUR PREVIOUS ACTIONS" is an action that YOU personally executed
- These are COMPLETED steps in your current task progression
- You MUST NOT repeat actions that are already completed
- You MUST build upon these completed steps to progress forward
- If you created a folder/file in previous actions, it now EXISTS and should be USED, not recreated

CRITICAL - EXTRACTED INFORMATION ANALYSIS:
{extracted_information}

PRIORITY ANALYSIS OF EXTRACTED DATA:
1. **IMMEDIATELY RELEVANT**: Filenames, URLs, error messages, confirmation codes - USE THESE NOW
2. **CONTEXTUALLY IMPORTANT**: UI element states, window titles, button labels - REFERENCE THESE
3. **PATTERNS TO FOLLOW**: Successful interaction methods, working coordinates - REPLICATE THESE
4. **AVOID THESE**: Failed approaches, broken elements, incorrect coordinates - DO NOT REPEAT

COORDINATES TO AVOID (previously failed):
{failure_coordinates}

INTELLIGENT ACTION SELECTION:
Based on the extracted information above:
- If you have extracted filenames/URLs, prioritize actions that use them
- If you have error messages, address the specific error mentioned
- If you have confirmation codes, use them in the next action
- If you have successful coordinates from similar actions, use those
- If previous attempts failed, analyze WHY and choose a fundamentally different approach

CRITICAL - AVOID REPEATING YOUR COMPLETED ACTIONS:
Analyze "YOUR PREVIOUS ACTIONS" to identify:
1. Actions that YOU have already completed successfully
2. Objects that YOU have already created (folders, files, etc.)
3. Steps that are already done and should not be repeated

If you detect that you are trying to repeat an action you already completed:
- STOP and recognize that the step is already done
- Use the object/result you created in the next step
- Move forward to the next logical step in the task
- If no progress is possible, output END to stop execution

Available Commands:
- click(x, y) - Click at normalized coordinates (0.0-1.0)
- double_click(x, y) - Double-click at normalized coordinates
- right_click(x, y) - Right-click at normalized coordinates
- text("content") - Input text content
- key(keys) - Press key combination (e.g., "ctrl+c", "enter", "cmd+space")
- drag(start_x, start_y, end_x, end_y) - Drag from start to end coordinates
- scroll(direction, amount) - Scroll (direction: up/down/left/right, amount: 1-10)
- END - End task execution

Use normalized coordinates (0.0-1.0). The top-left corner of the screen is (0.0, 0.0), the bottom-right corner is (1.0, 1.0).

Convert the task description into the appropriate command. Assess the situation and output only one optimal command to execute next.

IMPORTANT: You must use the following output format for every operation:
Line 1: Reasoning: [Why this action is being taken, considering YOUR previous actions AND extracted information]
Line 2: Target: [Specific target for the action]
Line 3: [The command to execute]
Line 4: save("[Content describing YOUR action and any useful information for future reference"])

SAVE COMMAND - YOUR ACTION LOG:
The save command is YOUR personal action log and reflection tool. Use it to:
- Record YOUR completed actions for future reference
- Preserve information about objects YOU created (folders, files, etc.)
- Document results of YOUR actions for the next step
- Store extracted information that YOU will need later
- Create a trail of YOUR progress through the task

This ensures YOU can track what YOU have already accomplished and avoid repeating completed steps.

Do not output descriptions; output only the formatted 4-line structure. Write only one command per line.

Example with learning from YOUR previous actions and extracted information:
Reasoning: Based on my previous action, I created 'ProjectFolder' which now exists. I need to open this folder I created to continue the task.
Target: ProjectFolder that I created in the previous step
click(0.4, 0.6)
save("Opened the ProjectFolder I created earlier, now ready for next step")

END""",
            
        }
    
    def get_template(self, task_type: TaskType) -> str:
        """Get prompt template for task type"""
        return self.templates.get(task_type.value, self.templates[TaskType.TASK_GENERATION.value])
    
    def format_prompt(self, task_type: TaskType, **kwargs) -> str:
        """Format prompt template with variables"""
        template = self.get_template(task_type)
        return template.format(**kwargs)


class ModelRunner:
    """2-Phase Architecture Model Runner: Ollama and Google Cloud Models"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Load base config
        self.config = config or load_config().api.__dict__
        self.logger = get_logger("model_runner")
        
        # Load settings manager to get API keys and preferences
        try:
            from ..utils.settings_manager import get_settings_manager
            settings_manager = get_settings_manager()
            
            # Override config with settings from settings manager
            self.config['google_api_key'] = settings_manager.get_google_api_key()
            self.config['preferred_provider'] = settings_manager.get_preferred_provider()
        except ImportError:
            self.logger.warning("Settings manager not available, using config only")
        
        # Initialize components
        self.vision_client = VisionAPIClient(self.config)
        self.prompt_template = PromptTemplate()
        
        self.logger.info(
            "Model runner initialized for 2-Phase Architecture",
            preferred_provider=self.config.get("preferred_provider", "ollama"),
            model=self.config.get("local_model", "gemini-3-flash-preview:latest"),
            google_api_configured=bool(self.config.get("google_api_key")),
        )
    
    def run_model(self, request: ModelRequest) -> ModelResponse:
        """Run AI model for 2-Phase Architecture"""
        start_time = time.time()
        
        try:
            # Validate request
            self._validate_request(request)
            
            # Format prompt
            prompt = self._format_prompt(request)
            
            # Create API request for AI model via selected provider
            # Determine provider based on configuration
            from .vision_api_client import APIProvider
            provider_enum = None
            model_name = None
            
            preferred_provider = self.config.get("preferred_provider", "ollama")
            if preferred_provider == "google" and self.config.get("google_api_key"):
                provider_enum = APIProvider.GOOGLE
                model_name = "gemini-3-flash-preview"  # Google's model
            else:
                provider_enum = APIProvider.OLLAMA
                model_name = self.config.get("local_model", "gemini-3-flash-preview:latest")
            
            api_request = APIRequest(
                prompt=prompt,
                image_data=request.image_data,
                image_format=request.image_format,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                model=model_name,
                provider=provider_enum,
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
        
        if request.max_tokens < 1 or request.max_tokens > 7000:
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
    


# Global model runner instance
_model_runner: Optional[ModelRunner] = None


def get_model_runner() -> ModelRunner:
    """Get global model runner instance"""
    global _model_runner
    
    if _model_runner is None:
        _model_runner = ModelRunner()
    
    return _model_runner
