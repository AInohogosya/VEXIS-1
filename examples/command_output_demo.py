#!/usr/bin/env python3
"""
Command Output Format Demonstration
Shows the new output format: reasoning -> target -> command -> save
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_agent.core_processing.command_output import format_click_command, format_command_output


def demo_command_output_format():
    """Demonstrate the new command output format"""
    print("=== Command Output Format Demo ===\n")
    
    print("1. Click command with reasoning and target:")
    output = format_click_command(
        reasoning="Need to access search functionality to find the required information",
        target="search bar at top of the page",
        x=0.5,
        y=0.2,
        save_content="Clicked search bar, ready to type query"
    )
    print(output)
    print()
    
    print("2. Another click command example:")
    output = format_click_command(
        reasoning="The submit button needs to be clicked to process the form",
        target="submit button at bottom right of form",
        x=0.8,
        y=0.7,
        save_content="Clicked submit button, form submitted successfully"
    )
    print(output)
    print()
    
    print("3. Text input command:")
    output = format_command_output(
        reasoning="Need to enter the search query to find relevant results",
        target="search input field",
        command="type_text('Python tutorial')",
        save_content="Typed search query: Python tutorial"
    )
    print(output)
    print()
    
    print("4. Failure example:")
    from ai_agent.core_processing.command_output import get_command_formatter
    formatter = get_command_formatter()
    
    output = formatter.format_failure_output(
        reasoning="Attempted to click on the button but it was not responsive",
        target="login button",
        command="click(0.3, 0.7)",
        error_message="Element not clickable - possibly disabled or hidden"
    )
    print(output)
    print()
    
    print("=== Format Specification ===")
    print("Line 1: Reasoning: Why this action is being taken")
    print("Line 2: Target: Specific target for the action")
    print("Line 3: Command: The actual command to execute")
    print("Line 4: save: The save command (always last line)")
    print()
    print("The save command is recognized simply as 'save' - not 'Save Command' or similar.")


if __name__ == "__main__":
    demo_command_output_format()
