"""
Comprehensive logging system for AI Agent
Zero-defect policy: detailed logging with structured output
"""

import sys
import logging
import structlog
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import json
import traceback

from .exceptions import AIAgentException


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'exc_info',
                'exc_text', 'stack_info'
            }:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class AIAgentLogger:
    """Enhanced logger for AI Agent with comprehensive features"""
    
    def __init__(
        self,
        name: str,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        enable_json: bool = False,
        enable_console: bool = True
    ):
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file
        self.enable_json = enable_json
        self.enable_console = enable_console
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_handlers()
        
        # Setup structlog processor
        self._setup_structlog()
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        formatter = JSONFormatter() if self.enable_json else logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _setup_structlog(self):
        """Setup structlog processor"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if self.enable_json else structlog.dev.ConsoleRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, extra=kwargs)
    
    def log_command(
        self,
        command: str,
        success: bool,
        duration: Optional[float] = None,
        error: Optional[str] = None,
        **kwargs
    ):
        """Log command execution"""
        self.info(
            "Command executed",
            command=command,
            success=success,
            duration=duration,
            error=error,
            **kwargs
        )
    
    def log_screenshot(
        self,
        screenshot_path: str,
        resolution: str,
        capture_method: str,
        success: bool,
        **kwargs
    ):
        """Log screenshot capture"""
        self.info(
            "Screenshot captured",
            screenshot_path=screenshot_path,
            resolution=resolution,
            capture_method=capture_method,
            success=success,
            **kwargs
        )
    
    def log_api_call(
        self,
        endpoint: str,
        method: str,
        status_code: Optional[int] = None,
        duration: Optional[float] = None,
        error: Optional[str] = None,
        **kwargs
    ):
        """Log API call"""
        self.info(
            "API call",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration=duration,
            error=error,
            **kwargs
        )
    
    def log_task_step(
        self,
        task_id: str,
        step: int,
        total_steps: int,
        action: str,
        success: bool,
        **kwargs
    ):
        """Log task step"""
        self.info(
            "Task step",
            task_id=task_id,
            step=step,
            total_steps=total_steps,
            action=action,
            success=success,
            **kwargs
        )
    
    def log_error_with_context(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log error with full context"""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        
        if isinstance(error, AIAgentException):
            error_info.update({
                "error_code": error.error_code,
                "context": error.context,
            })
        
        if context:
            error_info["context"] = context
        
        self.error(
            "Error occurred",
            **error_info
        )
    
    
    def log_command_generation(
        self,
        task_description: str,
        command: str,
        success: bool,
        model: str,
        latency: Optional[float] = None,
        **kwargs
    ):
        """Log command generation from AI"""
        self.info(
            "Command generated",
            event_type="command_generation",
            task_description=task_description,
            command=command,
            success=success,
            model=model,
            latency=latency,
            **kwargs
        )
    
    def log_task_execution(
        self,
        task_index: int,
        task_description: str,
        success: bool,
        commands_executed: int,
        duration: Optional[float] = None,
        **kwargs
    ):
        """Log task execution completion"""
        self.info(
            "Task execution completed",
            event_type="task_execution",
            task_index=task_index,
            task_description=task_description,
            success=success,
            commands_executed=commands_executed,
            duration=duration,
            **kwargs
        )


# Global logger registry
_loggers: Dict[str, AIAgentLogger] = {}


def get_logger(
    name: Optional[str] = None,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_json: Optional[bool] = None,
    enable_console: Optional[bool] = None
) -> AIAgentLogger:
    """Get or create logger instance"""
    if name is None:
        name = "ai_agent"
    
    if name not in _loggers:
        # Get default configuration
        from .config import load_config
        config = load_config()
        
        logger_config = config.logging.__dict__ if hasattr(config.logging, '__dict__') else {}
        
        _loggers[name] = AIAgentLogger(
            name=name,
            log_level=log_level or logger_config.get("level", "INFO"),
            log_file=log_file or logger_config.get("file"),
            enable_json=enable_json or logger_config.get("json_format", False),
            enable_console=enable_console if enable_console is not None else logger_config.get("console", True)
        )
    
    return _loggers[name]


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_json: bool = False,
    enable_console: bool = True
):
    """Setup global logging configuration"""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    formatter = JSONFormatter() if enable_json else logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


# Context manager for logging
class LogContext:
    """Context manager for adding logging context"""
    
    def __init__(self, logger: AIAgentLogger, **context):
        self.logger = logger
        self.context = context
    
    def __enter__(self):
        self.logger.info("Context entered", **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.error(
                "Context exited with error",
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                **self.context
            )
        else:
            self.logger.info("Context exited successfully", **self.context)
        return False  # Don't suppress exceptions
