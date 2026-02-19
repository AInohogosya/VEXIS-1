"""
Exception classes for AI Agent System
Minimal exceptions for 2-Phase Architecture
"""


class AIAgentException(Exception):
    """Base exception for AI Agent system"""
    pass


class APIError(AIAgentException):
    """API-related error"""
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class ValidationError(AIAgentException):
    """Validation error"""
    def __init__(self, message, field=None, value=None):
        super().__init__(message)
        self.field = field
        self.value = value


class ConfigurationError(AIAgentException):
    """Configuration error"""
    pass


class PlatformError(AIAgentException):
    """Platform-related error"""
    pass


class ScreenshotError(AIAgentException):
    """Screenshot-related error"""
    pass


class ExecutionError(AIAgentException):
    """Execution error"""
    pass


class TaskGenerationError(AIAgentException):
    """Task generation error"""
    def __init__(self, message, instruction=None):
        super().__init__(message)
        self.instruction = instruction


class CommandParsingError(AIAgentException):
    """Command parsing error"""
    pass


class VerificationError(AIAgentException):
    """Task verification error"""
    def __init__(self, message, task=None):
        super().__init__(message)
        self.task = task
