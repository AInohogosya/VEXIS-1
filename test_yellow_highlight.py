#!/usr/bin/env python3
"""
Test script to verify yellow highlight navigation works
"""

import sys
import os
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Add yellow-highlight-navigation to path
sys.path.insert(0, str(Path(__file__).parent / "yellow-highlight-navigation"))

from clean_interactive_menu import CleanInteractiveMenu

def test_yellow_highlight():
    """Test the yellow highlight menu"""
    menu = CleanInteractiveMenu(
        "Test Yellow Highlight",
        "This should show yellow highlight selection:"
    )
    
    menu.add_item("Option 1", "First option with yellow highlight", "option1", "🟡")
    menu.add_item("Option 2", "Second option with yellow highlight", "option2", "🟡")
    menu.add_item("Option 3", "Third option with yellow highlight", "option3", "🟡")
    
    print("Testing yellow highlight navigation...")
    print("Use arrow keys to navigate, Enter to select, 'q' to quit")
    
    result = menu.show()
    print(f"Selected: {result}")

if __name__ == "__main__":
    test_yellow_highlight()
