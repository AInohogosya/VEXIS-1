#!/usr/bin/env python3
"""
Ollama Error Handler for VEXIS-1.2
Specialized error handling for Ollama operations
Optimized for Gemini model management
"""

import re
import subprocess
from typing import Dict, Any, Optional, List
from ..utils.logger import get_logger
from .model_definitions import is_gemini_model, get_gemini_model_info


class OllamaError:
    """Represents an Ollama-specific error with context"""
    
    def __init__(self, error_type: str, message: str, suggestions: List[str] = None):
        self.error_type = error_type
        self.message = message
        self.suggestions = suggestions or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'error_type': self.error_type,
            'message': self.message,
            'suggestions': self.suggestions
        }


class OllamaErrorHandler:
    """Handles Ollama-specific errors with context-aware suggestions"""
    
    def __init__(self):
        self.logger = get_logger("ollama_error_handler")
        
        # Error patterns and their handlers
        self.error_patterns = {
            # Connection errors
            r'connection refused|connect.*failed|network.*unreachable': self._handle_connection_error,
            r'timeout|timed out': self._handle_timeout_error,
            
            # Model errors
            r'model.*not found|model.*does not exist': self._handle_model_not_found,
            r'model.*already exists|already installed': self._handle_model_exists,
            r'pull.*failed|download.*failed': self._handle_pull_failed,
            
            # Authentication errors
            r'not signed in|authentication.*failed|unauthorized': self._handle_auth_error,
            r'invalid.*credentials|api.*key.*invalid': self._handle_auth_error,
            
            # Permission errors
            r'permission denied|access denied': self._handle_permission_error,
            
            # Space errors
            r'no space|disk full|insufficient.*space': self._handle_space_error,
            
            # Version errors
            r'version.*mismatch|incompatible.*version': self._handle_version_error,
            
            # Cloud model specific errors
            r'cloud.*model|cloud.*access': self._handle_cloud_error,
            
            # Generic errors
            r'error|failed': self._handle_generic_error,
        }
    
    def handle_error(self, error_message: str, context: Dict[str, Any] = None, 
                   display_to_user: bool = False) -> OllamaError:
        """Handle an Ollama error with context-aware suggestions"""
        context = context or {}
        model_name = context.get('model_name', '')
        operation = context.get('operation', '')
        
        self.logger.debug(f"Handling Ollama error: {error_message}")
        
        # Try to match error patterns
        for pattern, handler in self.error_patterns.items():
            if re.search(pattern, error_message, re.IGNORECASE):
                error = handler(error_message, model_name, operation, context)
                break
        else:
            error = self._handle_generic_error(error_message, model_name, operation, context)
        
        # Log the error
        self.logger.error(f"Ollama error [{error.error_type}]: {error.message}")
        
        # Display to user if requested
        if display_to_user:
            self._display_error_to_user(error)
        
        return error
    
    def _handle_connection_error(self, message: str, model_name: str, operation: str, 
                               context: Dict[str, Any]) -> OllamaError:
        """Handle connection-related errors"""
        suggestions = [
            "Check if Ollama is running with: ollama --version",
            "Start Ollama service if needed",
            "Check network connectivity",
            "Verify Ollama is accessible at http://localhost:11434"
        ]
        
        if operation == 'pull_model':
            suggestions.extend([
                "Check internet connection for model download",
                "Try again with a different network"
            ])
        
        return OllamaError("connection_error", message, suggestions)
    
    def _handle_timeout_error(self, message: str, model_name: str, operation: str, 
                             context: Dict[str, Any]) -> OllamaError:
        """Handle timeout errors"""
        suggestions = [
            "Try the operation again",
            "Check network stability",
            "Consider using a smaller model for faster downloads"
        ]
        
        if operation == 'pull_model' and model_name:
            if is_gemini_model(model_name):
                model_info = get_gemini_model_info(model_name)
                suggestions.append(f"Try a lighter model like {model_info.get('name', 'gemma2:2b')}")
        
        suggestions.append("Increase timeout settings if the issue persists")
        
        return OllamaError("timeout_error", message, suggestions)
    
    def _handle_model_not_found(self, message: str, model_name: str, operation: str, 
                               context: Dict[str, Any]) -> OllamaError:
        """Handle model not found errors"""
        suggestions = [
            "Check the model name spelling",
            "Run 'ollama list' to see available models",
            "Pull the model first with: ollama pull <model_name>"
        ]
        
        if model_name and not is_gemini_model(model_name):
            # Suggest similar Gemini models
            suggestions.extend([
                "Consider using a Gemini model:",
                "  - gemma2:2b (lightweight and efficient)",
                "  - gemma3:4b (latest generation)",
                "  - gemini-3-flash-preview (cloud model)"
            ])
        elif model_name and is_gemini_model(model_name):
            model_info = get_gemini_model_info(model_name)
            suggestions.append(f"Install {model_info.get('name', model_name)} with: ollama pull {model_name}")
        
        return OllamaError("model_not_found", message, suggestions)
    
    def _handle_model_exists(self, message: str, model_name: str, operation: str, 
                           context: Dict[str, Any]) -> OllamaError:
        """Handle model already exists errors"""
        suggestions = [
            f"The model {model_name} is already installed",
            "Use 'ollama list' to see installed models",
            "Use a different model name if you want to install another model"
        ]
        
        if operation == 'pull_model':
            suggestions.append("Skip the pull step and use the existing model")
        
        return OllamaError("model_exists", message, suggestions)
    
    def _handle_pull_failed(self, message: str, model_name: str, operation: str, 
                          context: Dict[str, Any]) -> OllamaError:
        """Handle model pull failures"""
        suggestions = [
            "Check your internet connection",
            "Try pulling the model again",
            "Check available disk space"
        ]
        
        if model_name and is_gemini_model(model_name):
            model_info = get_gemini_model_info(model_name)
            suggestions.extend([
                f"Try a smaller Gemini model like gemma2:2b",
                "Check if the model name is correct for your region"
            ])
        
        if "cloud" in str(model_name).lower():
            suggestions.extend([
                "Ensure you're signed in with: ollama signin",
                "Check cloud model availability in your region",
                "Verify Ollama version supports cloud models"
            ])
        
        return OllamaError("pull_failed", message, suggestions)
    
    def _handle_auth_error(self, message: str, model_name: str, operation: str, 
                         context: Dict[str, Any]) -> OllamaError:
        """Handle authentication errors"""
        suggestions = [
            "Sign in to Ollama with: ollama signin",
            "Check your credentials",
            "Verify your Ollama account is active"
        ]
        
        if "cloud" in str(model_name).lower() or operation == 'cloud_model':
            suggestions.extend([
                "Cloud models require authentication",
                "Run 'ollama whoami' to check sign-in status",
                "Update Ollama to the latest version for cloud support"
            ])
        
        return OllamaError("auth_error", message, suggestions)
    
    def _handle_permission_error(self, message: str, model_name: str, operation: str, 
                               context: Dict[str, Any]) -> OllamaError:
        """Handle permission errors"""
        suggestions = [
            "Check file and directory permissions",
            "Run with appropriate user permissions",
            "Verify Ollama installation directory access"
        ]
        
        if operation == 'pull_model':
            suggestions.append("Check write permissions for Ollama model directory")
        
        return OllamaError("permission_error", message, suggestions)
    
    def _handle_space_error(self, message: str, model_name: str, operation: str, 
                          context: Dict[str, Any]) -> OllamaError:
        """Handle disk space errors"""
        suggestions = [
            "Free up disk space",
            "Remove unused models with: ollama rm <model_name>",
            "Check available space with: df -h"
        ]
        
        if model_name and is_gemini_model(model_name):
            suggestions.extend([
                "Try a smaller Gemini model:",
                "  - gemma3:1b (ultra lightweight)",
                "  - gemma2:2b (efficient)"
            ])
        
        return OllamaError("space_error", message, suggestions)
    
    def _handle_version_error(self, message: str, model_name: str, operation: str, 
                            context: Dict[str, Any]) -> OllamaError:
        """Handle version compatibility errors"""
        suggestions = [
            "Update Ollama to the latest version",
            "Check Ollama version with: ollama --version",
            "Reinstall Ollama if needed"
        ]
        
        if "cloud" in str(model_name).lower():
            suggestions.extend([
                "Cloud models require Ollama 0.17.0 or later",
                "Update with: curl -fsSL https://ollama.com/install.sh | sh"
            ])
        
        return OllamaError("version_error", message, suggestions)
    
    def _handle_cloud_error(self, message: str, model_name: str, operation: str, 
                          context: Dict[str, Any]) -> OllamaError:
        """Handle cloud model specific errors"""
        suggestions = [
            "Ensure Ollama version supports cloud models (0.17.0+)",
            "Sign in with: ollama signin",
            "Check cloud model availability in your region",
            "Verify internet connectivity"
        ]
        
        if model_name and is_gemini_model(model_name):
            model_info = get_gemini_model_info(model_name)
            if model_info.get('type') == 'cloud':
                suggestions.extend([
                    f"Cloud model {model_info.get('name', model_name)} requires authentication",
                    "Try a local Gemini model as alternative",
                    "Check if cloud services are available in your region"
                ])
        
        return OllamaError("cloud_error", message, suggestions)
    
    def _handle_generic_error(self, message: str, model_name: str, operation: str, 
                            context: Dict[str, Any]) -> OllamaError:
        """Handle generic errors"""
        suggestions = [
            "Check Ollama service status",
            "Verify the command syntax",
            "Try the operation again",
            "Check Ollama logs for more details"
        ]
        
        if model_name:
            suggestions.append(f"Verify model name: {model_name}")
        
        if operation:
            suggestions.append(f"Operation: {operation}")
        
        return OllamaError("generic_error", message, suggestions)
    
    def _display_error_to_user(self, error: OllamaError):
        """Display error to user in a user-friendly format"""
        print(f"\n❌ Ollama Error: {error.error_type}")
        print(f"📝 {error.message}")
        
        if error.suggestions:
            print("\n💡 Suggestions:")
            for i, suggestion in enumerate(error.suggestions, 1):
                print(f"   {i}. {suggestion}")
        
        print("\n" + "="*50)
    
    def get_error_summary(self, errors: List[OllamaError]) -> Dict[str, Any]:
        """Get summary of multiple errors"""
        error_types = {}
        for error in errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        return {
            'total_errors': len(errors),
            'error_types': error_types,
            'most_common': max(error_types.items(), key=lambda x: x[1]) if error_types else None
        }


# Global error handler instance
_error_handler = None


def get_error_handler() -> OllamaErrorHandler:
    """Get the global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = OllamaErrorHandler()
    return _error_handler


def handle_ollama_error(error_message: str, context: Dict[str, Any] = None, 
                       display_to_user: bool = False) -> OllamaError:
    """Convenience function to handle Ollama errors"""
    handler = get_error_handler()
    return handler.handle_error(error_message, context, display_to_user)


def handle_subprocess_error(result: subprocess.CompletedProcess, context: Dict[str, Any] = None,
                          display_to_user: bool = False) -> OllamaError:
    """Handle subprocess execution errors"""
    error_message = result.stderr or result.stdout or "Unknown error"
    if context is None:
        context = {}
    
    context['return_code'] = result.returncode
    
    return handle_ollama_error(error_message, context, display_to_user)


if __name__ == "__main__":
    # Test error handler
    handler = OllamaErrorHandler()
    
    test_errors = [
        ("connection refused", {"operation": "pull_model", "model_name": "gemma2:2b"}),
        ("model not found: invalid_model", {"operation": "pull_model", "model_name": "invalid_model"}),
        ("not signed in", {"operation": "cloud_model", "model_name": "gemini-3-flash-preview"}),
        ("no space left on device", {"operation": "pull_model", "model_name": "gemma3:27b"}),
    ]
    
    for error_msg, context in test_errors:
        error = handler.handle_error(error_msg, context, display_to_user=True)
        print(f"\nError type: {error.error_type}")
        print(f"Suggestions: {len(error.suggestions)}")
