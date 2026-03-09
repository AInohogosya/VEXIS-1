#!/usr/bin/env python3
"""
Test script to verify provider selection with yellow highlight
"""

import sys
import os
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Add yellow-highlight-navigation to path
sys.path.insert(0, str(Path(__file__).parent / "yellow-highlight-navigation"))

from ai_agent.utils.config_manager import ConfigManager

def test_provider_selection():
    """Test the provider selection with yellow highlight"""
    try:
        config_manager = ConfigManager()
        print("Testing provider selection with yellow highlight...")
        print("Use arrow keys to navigate, Enter to select, 'q' to quit")
        
        # This should show the yellow highlight menu
        selected_provider = config_manager.select_provider()
        print(f"Selected provider: {selected_provider}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_provider_selection()
