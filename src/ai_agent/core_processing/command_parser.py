"""
Command Parser for AI Agent System
2-Phase Vision-Only Architecture: Strict Command Set Enforcement
"""

import re
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from ..external_integration.model_runner import get_model_runner
# Simple exception classes
class CommandParsingError(Exception):
    """Simple command parsing error"""
    pass

class ValidationError(Exception):
    """Simple validation error"""
    pass


class CommandType(Enum):
    """Strict command set for 2-Phase Architecture"""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TEXT = "text"
    KEY = "key"
    DRAG = "drag"
    SCROLL = "scroll"
    END = "end"


@dataclass
class ParsedCommand:
    """Parsed command structure"""
    type: CommandType
    parameters: Dict[str, Any]
    raw_text: str


@dataclass
class Coordinate:
    """Coordinate structure"""
    x: float
    y: float
    
    def __post_init__(self):
        # Validate coordinates are in normalized range (0.0-1.0)
        if not (0.0 <= self.x <= 1.0):
            raise ValidationError(f"Invalid x coordinate: {self.x}. Must be between 0.0 and 1.0")
        if not (0.0 <= self.y <= 1.0):
            raise ValidationError(f"Invalid y coordinate: {self.y}. Must be between 0.0 and 1.0")
    
    def to_tuple(self) -> tuple:
        return (self.x, self.y)


class CommandParser:
    """Simplified command parser for 2-Phase Vision-Only Architecture"""
    
    def __init__(self):
        # Strict command patterns
        self.command_patterns = self._initialize_strict_command_patterns()
    
    def _initialize_strict_command_patterns(self) -> Dict[CommandType, list]:
        """Initialize strict command patterns for 2-Phase Architecture"""
        return {
            CommandType.CLICK: [
                re.compile(r'^click\s*\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)$', re.IGNORECASE),
            ],
            CommandType.DOUBLE_CLICK: [
                re.compile(r'^double_click\s*\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)$', re.IGNORECASE),
            ],
            CommandType.RIGHT_CLICK: [
                re.compile(r'^right_click\s*\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)$', re.IGNORECASE),
            ],
            CommandType.TEXT: [
                re.compile(r'^text\s*\(\s*["\']([^"\']*)["\']\s*\)$', re.IGNORECASE),
            ],
            CommandType.KEY: [
                re.compile(r'^key\s*\(\s*["\']([^"\']*)["\']\s*\)$', re.IGNORECASE),
            ],
            CommandType.DRAG: [
                re.compile(r'^drag\s*\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)$', re.IGNORECASE),
            ],
            CommandType.SCROLL: [
                re.compile(r'^scroll\s*\(\s*["\']?(up|down|left|right)["\']?\s*,\s*([0-9]+)\s*\)$', re.IGNORECASE),
            ],
            CommandType.END: [
                re.compile(r'^end\s*$', re.IGNORECASE),
            ],
        }
    
    
    def parse_command(self, command_text: str, screenshot: Optional[bytes] = None, context: Optional[Dict[str, Any]] = None) -> ParsedCommand:
        """Parse command text into structured command using AI vision analysis"""
        # Validate input first
        if command_text is None:
            raise ValidationError("Command text cannot be None", "command_text", command_text)
        
        if not command_text or not command_text.strip():
            raise ValidationError("Command text cannot be empty", "command_text", command_text)
        
        try:
            # Apply rule: Read from second line downwards
            lines = command_text.strip().split('\n')
            if len(lines) >= 2:
                # Extract second line from bottom (click/text command)
                second_from_bottom = lines[-2].strip()
                # Extract final line (save command)
                final_line = lines[-1].strip()
                
                # Parse the main command from second line from bottom
                cleaned_command = self._clean_command_text(second_from_bottom)
                parsed = self._parse_with_patterns(cleaned_command)
                
                if parsed:
                    return parsed
            else:
                # Fallback to original parsing for single line commands
                cleaned_command = self._clean_command_text(command_text)
                parsed = self._parse_with_patterns(cleaned_command)
                
                if parsed:
                    return parsed
            
            # For completely invalid commands, raise validation error
            raise ValidationError(f"Invalid command format: {command_text}", "command_text", command_text)
            
        except ValidationError:
            # Re-raise validation errors - these are legitimate failures
            raise
        except Exception as e:
            # For other exceptions, fail the command
            raise CommandParsingError(f"Failed to parse command: {command_text}")
    
    def _clean_command_text(self, command_text: str) -> str:
        """Clean and normalize command text"""
        if not command_text:
            raise ValidationError("Command text cannot be empty", "command_text", command_text)
        
        # Remove extra whitespace but preserve structure for validation
        cleaned = ' '.join(command_text.split())
        
        # Remove common prefixes
        prefixes_to_remove = ["please ", "can you ", "i want you to ", "now "]
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):]
        
        final_cleaned = cleaned.strip()
        
        # Final validation
        if not final_cleaned:
            raise ValidationError("Command text cannot be empty after cleaning", "command_text", command_text)
        
        return final_cleaned
    
        
    def _parse_with_patterns(self, command_text: str) -> Optional[ParsedCommand]:
        """Parse command using strict regex patterns"""
        command_lower = command_text.lower()
        
        # Try each command type
        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = pattern.match(command_text)
                if match:
                    return self._create_command_from_match(command_type, match, command_text)
        
        return None
    
    def _create_command_from_match(self, command_type: CommandType, match: re.Match, raw_text: str) -> ParsedCommand:
        """Create parsed command from regex match with strict validation"""
        try:
            parameters = {}
            
            if command_type in [CommandType.CLICK, CommandType.DOUBLE_CLICK, CommandType.RIGHT_CLICK]:
                # Click commands: (x, y)
                x, y = float(match.group(1)), float(match.group(2))
                parameters = {"coordinates": Coordinate(x, y)}
                
            elif command_type == CommandType.TEXT:
                # Text command: ("content")
                text_content = match.group(1)
                if not text_content:
                    raise ValidationError("Text content cannot be empty")
                parameters = {"text": text_content}
                
            elif command_type == CommandType.KEY:
                # Key command: ("keys")
                key_combo = match.group(1)
                if not key_combo:
                    raise ValidationError("Key combination cannot be empty")
                parameters = {"keys": key_combo}
                
            elif command_type == CommandType.DRAG:
                # Drag command: (start_x, start_y, end_x, end_y)
                start_x, start_y = float(match.group(1)), float(match.group(2))
                end_x, end_y = float(match.group(3)), float(match.group(4))
                parameters = {
                    "start": Coordinate(start_x, start_y),
                    "end": Coordinate(end_x, end_y)
                }
                
            elif command_type == CommandType.SCROLL:
                # Scroll command: (direction, amount)
                direction = match.group(1)
                amount = int(match.group(2))
                if direction not in ["up", "down", "left", "right"]:
                    raise ValidationError(f"Invalid scroll direction: {direction}")
                if not (1 <= amount <= 10):
                    raise ValidationError(f"Invalid scroll amount: {amount}. Must be between 1 and 10")
                parameters = {"direction": direction, "amount": amount}
                
            elif command_type == CommandType.END:
                # END command: no parameters
                parameters = {}
            
            return ParsedCommand(
                type=command_type,
                parameters=parameters,
                raw_text=raw_text,
            )
            
        except Exception as e:
            return None
