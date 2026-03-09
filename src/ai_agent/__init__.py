"""
AI Agent System - Vision-based GUI automation
Production-ready implementation with zero-defect policy

Version Note: VEXIS-1.2 uses simplified versioning (1.2 instead of 1.2.0.1)
with 0.0.1 increments omitted to reduce complexity while maintaining compatibility.
"""

__version__ = "1.2"
__author__ = "AI Agent Team"
__email__ = "contact@vexis-1.2.org"
__description__ = "Vision-based AI agent for GUI automation"

# Import core components for easy access
from .core_processing.command_parser import CommandParser
from .core_processing.two_phase_engine import TwoPhaseEngine
from .platform_abstraction.screenshot_capture import ScreenshotCapture
from .platform_abstraction.gui_automation import GUIAutomation
from .external_integration.vision_api_client import VisionAPIClient
from .external_integration.model_runner import ModelRunner

__all__ = [
    "CommandParser", 
    "TwoPhaseEngine",
    "ScreenshotCapture",
    "GUIAutomation",
    "VisionAPIClient",
    "ModelRunner",
]
