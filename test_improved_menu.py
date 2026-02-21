#!/usr/bin/env python3
"""
Standalone test for the improved interactive menu
"""

import sys
import os
from pathlib import Path

# Add src to path for direct import
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Import only the interactive menu module directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "interactive_menu", 
    src_dir / "ai_agent" / "utils" / "interactive_menu.py"
)
interactive_menu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interactive_menu)

def test_menu():
    """Test the improved interactive menu"""
    menu = interactive_menu.InteractiveMenu(
        'Test Menu - Rich Live Display', 
        'Testing smooth navigation without flickering'
    )
    
    menu.add_item(
        'Option 1 - Speed Test', 
        'First option with description', 
        'opt1', 
        '🚀'
    )
    menu.add_item(
        'Option 2 - Quality Test', 
        'Second option with description', 
        'opt2', 
        '⭐'
    )
    menu.add_item(
        'Option 3 - Balance Test', 
        'Third option with description', 
        'opt3', 
        '💫'
    )
    
    print('🎯 Testing improved menu navigation:')
    print('   Use ↑/↓ arrows to navigate (should be smooth)')
    print('   Press Enter to select, q to quit')
    print()
    
    result = menu.show()
    print(f'\n✅ Final result: {result}')
    return result

if __name__ == "__main__":
    test_menu()
