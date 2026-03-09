#!/usr/bin/env python3
"""
Keyboard Mapping Demo
Demonstrates OS-specific keyboard support
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_agent.platform_abstraction.keyboard_mapping import get_keyboard_mapping, KeyCategory
from ai_agent.platform_abstraction.gui_automation import GUIAutomation


def main():
    print("=== Keyboard Mapping Demo ===\n")
    
    # Initialize keyboard mapping
    keyboard_mapping = get_keyboard_mapping()
    
    # Get keyboard info
    keyboard_info = keyboard_mapping.get_keyboard_info()
    print(f"OS: {keyboard_info['os']}")
    print(f"Total keys: {keyboard_info['total_keys']}")
    print(f"OS-specific keys: {keyboard_info['os_specific_keys']}")
    print(f"Common keys: {keyboard_info['common_keys']}")
    print()
    
    # Show available keys
    print("Available keys:")
    for key in keyboard_info['available_keys']:
        print(f"  - {key}")
    print()
    
    # Show categories
    print("Key categories:")
    for category, count in keyboard_info['categories'].items():
        print(f"  - {category}: {count} keys")
    print()
    
    # Show OS-specific keys
    print("OS-specific keys:")
    os_specific_keys = keyboard_mapping.get_os_specific_keys()
    for key in os_specific_keys:
        print(f"  - {key.name}: {key.description}")
    print()
    
    # Test key combinations
    print("Testing key combinations:")
    test_combinations = []
    
    if keyboard_info['os'] == 'darwin':
        test_combinations = [
            "cmd+c",      # Copy
            "cmd+v",      # Paste
            "cmd+shift+4", # Screenshot
            "cmd+space",  # Spotlight
        ]
    elif keyboard_info['os'] == 'windows':
        test_combinations = [
            "ctrl+c",     # Copy
            "ctrl+v",     # Paste
            "win+e",      # File Explorer
            "win+l",      # Lock screen
        ]
        if 'copilot' in keyboard_info['available_keys']:
            test_combinations.append("copilot")  # Copilot key
    elif keyboard_info['os'] == 'linux':
        test_combinations = [
            "ctrl+c",     # Copy
            "ctrl+v",     # Paste
            "super+t",    # New terminal (common shortcut)
            "alt+tab",    # Switch windows
        ]
    
    for combo in test_combinations:
        normalized = keyboard_mapping.normalize_key_combination(combo)
        print(f"  {combo} -> {normalized}")
    print()
    
    # Initialize GUI automation
    print("Initializing GUI automation...")
    automation = GUIAutomation()
    
    # Show available keys through automation
    available_keys = automation.get_available_keys()
    print(f"Detected {len(available_keys)} available keys")
    
    # Test some key info
    print("\nKey information examples:")
    test_keys = ['cmd', 'win', 'super', 'ctrl', 'alt', 'copilot']
    
    for key in test_keys:
        info = automation.get_key_info(key)
        if info:
            print(f"  {key}:")
            print(f"    Name: {info['name']}")
            print(f"    PyAutoGUI key: {info['pyautogui_key']}")
            print(f"    Category: {info['category']}")
            print(f"    OS-specific: {info['os_specific']}")
            if info['alternatives']:
                print(f"    Alternatives: {', '.join(info['alternatives'])}")
            print(f"    Available: {automation.is_key_available(key)}")
            print()
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
