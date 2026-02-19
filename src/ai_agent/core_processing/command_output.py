"""
Command Output System for Phase 2: Execution Engine
Handles reasoning output and command formatting according to specifications:
1. Output reasoning first
2. Output specific target for clicking
3. Output click command (second-to-last line)
4. Output save command (final line)
"""

import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from ..utils.logger import get_logger
from .save_command import get_save_command, SaveContentType


@dataclass
class CommandOutput:
    """Structure for formatted command output"""
    reasoning: str
    target: str
    command: str
    save_content: str


class CommandOutputFormatter:
    """
    Formats command output according to specifications:
    1. Reasoning about the action
    2. Specific target for clicking
    3. Click command (second-to-last line)
    4. save command (final line)
    """
    
    def __init__(self):
        self.logger = get_logger("command_output")
        self.save_command = get_save_command()
    
    def format_command_output(
        self,
        reasoning: str,
        target: str,
        command: str,
        save_content: str,
        coordinates: Optional[Tuple[float, float]] = None,
        **save_kwargs
    ) -> str:
        """
        Format command output according to specifications
        
        Args:
            reasoning: Reasoning about why this action is being taken
            target: Specific target for clicking (e.g., "search bar at top right")
            command: The actual command to execute (e.g., "click(0.5, 0.2)")
            save_content: Content for the save command
            coordinates: Coordinates for the save command metadata
            **save_kwargs: Additional parameters for save command
        
        Returns:
            str: Formatted multi-line output
        """
        # Create the formatted output
        output_lines = [
            f"Reasoning: {reasoning}",
            f"Target: {target}",
            f"{command}",
            f'save("{save_content}")'
        ]
        
        # Join with newlines
        formatted_output = "\n".join(output_lines)
        
        # Save the command execution
        self.save_command.save(
            content=save_content,
            operation_command=command,
            coordinates=coordinates,
            visual_feedback=f"Targeted: {target}",
            metadata={
                "reasoning": reasoning,
                "target": target
            },
            content_type=SaveContentType.FEEDBACK,
            **save_kwargs
        )
        
        self.logger.debug(f"Formatted command output with reasoning and target")
        return formatted_output
    
    def format_click_command(
        self,
        reasoning: str,
        target: str,
        x: float,
        y: float,
        save_content: Optional[str] = None,
        **save_kwargs
    ) -> str:
        """
        Format a click command with reasoning and target
        
        Args:
            reasoning: Reasoning about the click action
            target: Specific target for clicking
            x: X coordinate (normalized 0.0-1.0)
            y: Y coordinate (normalized 0.0-1.0)
            save_content: Content for save command (auto-generated if None)
            **save_kwargs: Additional parameters for save command
        
        Returns:
            str: Formatted multi-line output
        """
        command = f"click({x}, {y})"
        
        if save_content is None:
            save_content = f"Clicked {target}. Response received."
        
        coordinates = (x, y) if save_kwargs.get('coordinates') is None else save_kwargs['coordinates']
        
        return self.format_command_output(
            reasoning=reasoning,
            target=target,
            command=command,
            save_content=save_content,
            coordinates=coordinates,
            **save_kwargs
        )
    
    def format_failure_output(
        self,
        reasoning: str,
        target: str,
        command: str,
        error_message: str,
        coordinates: Optional[Tuple[float, float]] = None,
        **save_kwargs
    ) -> str:
        """
        Format a failure output with reasoning and target
        
        Args:
            reasoning: Reasoning about why the action failed
            target: Target that was attempted
            command: The command that failed
            error_message: Description of the failure
            coordinates: Coordinates that failed
            **save_kwargs: Additional parameters for save command
        
        Returns:
            str: Formatted multi-line output
        """
        save_content = f"Failed to {target}: {error_message}"
        
        # Update save kwargs for failure
        save_kwargs.update({
            'content_type': SaveContentType.FAILURE,
            'failure_details': {
                'error': error_message,
                'reasoning': reasoning,
                'target': target
            }
        })
        
        return self.format_command_output(
            reasoning=reasoning,
            target=target,
            command=command,
            save_content=save_content,
            coordinates=coordinates,
            **save_kwargs
        )
    
    def format_extraction_output(
        self,
        reasoning: str,
        target: str,
        extracted_info: Dict[str, Any],
        save_content: Optional[str] = None,
        **save_kwargs
    ) -> str:
        """
        Format an extraction output with reasoning and target
        
        Args:
            reasoning: Reasoning about the extraction
            target: Target of extraction (e.g., "error message dialog")
            extracted_info: Dictionary of extracted information
            save_content: Content for save command (auto-generated if None)
            **save_kwargs: Additional parameters for save command
        
        Returns:
            str: Formatted multi-line output
        """
        if save_content is None:
            info_summary = ", ".join([f"{k}: {v}" for k, v in extracted_info.items()])
            save_content = f"Extracted from {target}: {info_summary}"
        
        # Update save kwargs for extraction
        save_kwargs.update({
            'content_type': SaveContentType.EXTRACTION,
            'extracted_info': extracted_info
        })
        
        # For extraction, we might not have a click command
        command = save_kwargs.get('operation_command', 'extract()')
        
        return self.format_command_output(
            reasoning=reasoning,
            target=target,
            command=command,
            save_content=save_content,
            **save_kwargs
        )


# Global formatter instance
_global_formatter: Optional[CommandOutputFormatter] = None


def get_command_formatter() -> CommandOutputFormatter:
    """Get global command formatter instance"""
    global _global_formatter
    if _global_formatter is None:
        _global_formatter = CommandOutputFormatter()
    return _global_formatter


def format_click_command(
    reasoning: str,
    target: str,
    x: float,
    y: float,
    save_content: Optional[str] = None,
    **kwargs
) -> str:
    """
    Global function to format click command with reasoning and target
    
    Example:
    format_click_command(
        reasoning="Need to access search functionality",
        target="search bar at top of page",
        x=0.5, y=0.2,
        save_content="Clicked search bar, ready to type query"
    )
    """
    return get_command_formatter().format_click_command(
        reasoning, target, x, y, save_content, **kwargs
    )


def format_command_output(
    reasoning: str,
    target: str,
    command: str,
    save_content: str,
    **kwargs
) -> str:
    """
    Global function to format any command with reasoning and target
    """
    return get_command_formatter().format_command_output(
        reasoning, target, command, save_content, **kwargs
    )
