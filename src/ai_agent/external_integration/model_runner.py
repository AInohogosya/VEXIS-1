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
            TaskType.TASK_GENERATION.value: """You are an AI assistant that analyzes user instructions and screen state to create a step-by-step task list.

User Instruction: {instruction}

Current Screen: [Screenshot provided]

Generate a numbered list of specific, sequential tasks to complete the instruction. Each task should:
- Be a single, actionable step
- Be atomic (can be done in one action)
- Consider the current screen state

Break complex actions into smaller steps. Example: Instead of "Open browser and search", use:
1. Click browser icon
2. Wait for browser to open  
3. Click search bar
4. Type search query
5. Press Enter

Output format:
1. [First task]
2. [Second task]
3. [Third task]
...

Provide only the numbered list.""",
            
            TaskType.COMMAND_PARSING.value: """You are an AI assistant that converts task descriptions into GUI automation commands.

Task: {task_description}

Current Screen: [Screenshot provided]
Previous Screen: [previous screenshot provided]
Previous Command: {previous_command}

YOUR COMPLETED ACTIONS:
{previous_save_content}

EXTRACTED INFORMATION:
{extracted_information}

FAILED COORDINATES TO AVOID:
{failure_coordinates}

KEY GUIDELINES:
- Build upon YOUR completed actions - don't repeat what you've already done
- Use extracted information (filenames, URLs, error messages) immediately
- Avoid previously failed coordinates
- Each command should be a single, specific action

Available Commands:
- click(x, y) - Click at normalized coordinates (0.0-1.0)
- double_click(x, y) - Double-click
- right_click(x, y) - Right-click  
- text("content") - Type text
- key(keys) - Press keys (e.g., "ctrl+c", "enter")
- drag(start_x, start_y, end_x, end_y) - Drag
- scroll(direction, amount) - Scroll (direction: up/down/left/right, amount: 1-10)
- END - End task execution

OUTPUT FORMAT (exactly 4 lines):
Line 1: Reasoning: [Why this action, considering your previous actions and extracted info]
Line 2: Target: [Specific target for the action]  
Line 3: [The command to execute]
Line 4: save("[Description of your action and useful information for future steps]")

Example:
Reasoning: I created 'ProjectFolder' in my previous action, so now I need to open it.
Target: ProjectFolder I created earlier
click(0.4, 0.6)
save("Opened the ProjectFolder I created, ready for next step")

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
                # Use the selected Google model from settings
                from ..utils.settings_manager import get_settings_manager
                settings_manager = get_settings_manager()
                model_name = settings_manager.get_google_model()
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
