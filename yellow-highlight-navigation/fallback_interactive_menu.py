"""
Fallback Menu System - Compatible with terminals that don't support cursor positioning
Uses screen clearing instead of in-place updates for maximum compatibility
"""

import sys
import os
from typing import Optional, List, Any
try:
    from .config import get_colors, get_navigation_config, get_display_config
except ImportError:
    # Fallback for direct execution - use the local config
    try:
        from config import get_colors, get_navigation_config, get_display_config
    except ImportError:
        # Ultimate fallback - define colors inline
        def get_colors():
            return {
                'RESET': '\033[0m',
                'BOLD': '\033[1m',
                'BLACK': '\033[30m',
                'YELLOW': '\033[33m',
                'BRIGHT_YELLOW': '\033[93m',
                'BG_YELLOW': '\033[43m',
                'WHITE': '\033[97m',
                'BRIGHT_WHITE': '\033[37m',
                'BRIGHT_GREEN': '\033[92m',
                'RED': '\033[91m',
                'CYAN': '\033[36m',
                'BRIGHT_CYAN': '\033[96m'
            }
        def get_navigation_config():
            return {'navigation': {}, 'arrow_keys': {}}
        def get_display_config():
            return {'display': {}}


class Colors:
    """Color constants from reproducible configuration"""
    def __init__(self):
        colors = get_colors()
        for key, value in colors.items():
            setattr(self, key, value)
    
    RESET = get_colors()['RESET']
    BOLD = get_colors()['BOLD']
    CYAN = get_colors()['CYAN']
    BRIGHT_CYAN = get_colors()['BRIGHT_CYAN']
    BRIGHT_WHITE = get_colors()['BRIGHT_WHITE']
    BRIGHT_YELLOW = get_colors()['BRIGHT_YELLOW']
    BRIGHT_GREEN = get_colors()['BRIGHT_GREEN']
    RED = get_colors()['RED']
    YELLOW = get_colors()['YELLOW']
    BG_YELLOW = get_colors()['BG_YELLOW']
    WHITE = get_colors()['WHITE']
    BLACK = get_colors()['BLACK']


# Initialize colors from config
_colors = Colors()
RESET = _colors.RESET
BOLD = _colors.BOLD
CYAN = _colors.CYAN
BRIGHT_CYAN = _colors.BRIGHT_CYAN
BRIGHT_WHITE = _colors.BRIGHT_WHITE
BRIGHT_YELLOW = _colors.BRIGHT_YELLOW
BRIGHT_GREEN = _colors.BRIGHT_GREEN
RED = _colors.RED
YELLOW = _colors.YELLOW
BG_YELLOW = _colors.BG_YELLOW
WHITE = _colors.WHITE
BLACK = _colors.BLACK


class FallbackInteractiveMenu:
    """Fallback menu that uses screen clearing for maximum compatibility"""
    
    def __init__(self, title: str, description: str):
        self.title = title
        self.description = description
        self.items = []
        self.current_index = 0
        
    def add_item(self, display_name: str, description: str, value: Any, icon: str = "📋"):
        self.items.append({
            "display_name": display_name,
            "description": description,
            "value": value,
            "icon": icon
        })
    
    def get_key(self) -> str:
        """Get key press with universal arrow key detection"""
        try:
            # Universal detection that works in all terminals
            return self._universal_get_key()
        except Exception:
            # Fallback to simple input
            return self._fallback_input()
    
    def _universal_get_key(self) -> str:
        """Universal key detection that works anywhere"""
        try:
            import select
            
            # Use select with timeout - most reliable method
            if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                ch = sys.stdin.read(1)
                
                if ch == '\x1b':
                    # Try to read the full escape sequence
                    seq = ch
                    for _ in range(3):  # Try to read up to 3 more chars
                        if select.select([sys.stdin], [], [], 0.01) == ([sys.stdin], [], []):
                            seq += sys.stdin.read(1)
                        else:
                            break
                    
                    # Normalize all possible arrow sequences
                    if seq in ['\x1b[A', '\x1bOA', '\x1b[A\x1bOA']:  # Up variations
                        return '\x1b[A'
                    elif seq in ['\x1b[B', '\x1bOB', '\x1b[B\x1bOB']:  # Down variations
                        return '\x1b[B'
                    elif seq in ['\x1b[C', '\x1bOC', '\x1b[C\x1bOC']:  # Right variations
                        return '\x1b[C'
                    elif seq in ['\x1b[D', '\x1bOD', '\x1b[D\x1bOD']:  # Left variations
                        return '\x1b[D'
                    else:
                        return seq  # Return other sequences as-is
                
                elif ch in ['\r', '\n']:
                    return '\r'
                elif ch.lower() in ['q', 'Q']:
                    return 'q'
                elif ch.isdigit():
                    return ch
                elif ch and len(ch) == 1:
                    return ch
            
            return ''
            
        except Exception:
            return ''
    
    def _fallback_input(self) -> str:
        """Fallback to regular input() method"""
        try:
            choice = input(f"{BRIGHT_YELLOW}Enter choice (1-{len(self.items)}) or 'q': {RESET}").strip()
            if choice.lower() == 'q':
                return 'q'
            elif choice.isdigit() and 1 <= int(choice) <= len(self.items):
                return choice
            else:
                return ''  # Invalid input
        except:
            return ''
    
    def display_menu(self):
        """Display the complete menu with current selection highlighted"""
        # Clear screen
        print("\033[2J\033[H", end="", flush=True)
        
        # Display header
        print(f"{BOLD}{BRIGHT_CYAN}🔧 {self.title}{RESET}")
        print(f"{BRIGHT_CYAN}{'─' * 50}{RESET}")
        print(f"{CYAN}📝 {self.description}{RESET}")
        print()
        print(f"{BOLD}📋 Available Options:{RESET}")
        print()
        
        # Display items with current selection highlighted
        for i, item in enumerate(self.items):
            if i == self.current_index:
                # Highlighted selection with yellow background
                print(f"{BG_YELLOW}{BLACK}  ▶ [{i+1}] {item['icon']} {item['display_name']}{RESET}")
                print(f"{BG_YELLOW}{BLACK}       {item['description']}{RESET}")
            else:
                # Normal option
                print(f"  [{i+1}] {item['icon']} {item['display_name']}")
                print(f"       {item['description']}")
            print()
        
        # Display footer
        print(f"{BRIGHT_CYAN}{'─' * 50}{RESET}")
        print(f"{BRIGHT_YELLOW}💡 Use ↑↓ arrows • 1-9 numbers • Enter to select • 'q' to quit{RESET}")
        print()
        print()
    
    def show(self) -> Optional[Any]:
        """Show interactive menu with fallback display updates"""
        try:
            # Initial display
            self.display_menu()
            
            while True:
                key = self.get_key()
                
                # Skip empty keys (no input)
                if not key or key == '':
                    continue
                
                # Handle input with improved key detection
                if isinstance(key, str) and key.lower() == 'q':
                    # Clear screen before exit
                    print("\033[2J\033[H", end="", flush=True)
                    return None
                elif key in ['\r', '\n']:  # Enter key
                    selected_item = self.items[self.current_index]
                    # Clear screen before return
                    print("\033[2J\033[H", end="", flush=True)
                    return selected_item["value"]
                elif key == '\x1b[A':  # Up arrow
                    old_index = self.current_index
                    self.current_index = max(0, self.current_index - 1)
                    if old_index != self.current_index:
                        self.display_menu()  # Redisplay entire menu
                elif key == '\x1b[B':  # Down arrow
                    old_index = self.current_index
                    self.current_index = min(len(self.items) - 1, self.current_index + 1)
                    if old_index != self.current_index:
                        self.display_menu()  # Redisplay entire menu
                elif isinstance(key, str) and key.isdigit():  # Number input
                    try:
                        num = int(key)
                        if 1 <= num <= len(self.items):
                            old_index = self.current_index
                            self.current_index = num - 1
                            if old_index != self.current_index:
                                self.display_menu()  # Redisplay entire menu
                    except:
                        pass
                    
        except KeyboardInterrupt:
            print("\033[2J\033[H", end="", flush=True)
            return None
        except Exception:
            return self.fallback_selection()
    
    def fallback_selection(self) -> Optional[Any]:
        """Fallback to numbered selection with clean display"""
        print(f"\n{RED}⚠️  Using simple selection as fallback{RESET}")
        print()
        
        for i, item in enumerate(self.items, 1):
            print(f"{i}. {item['icon']} {item['display_name']}")
            print(f"   {item['description']}")
            print()
        
        while True:
            try:
                choice = input(f"{YELLOW}Select (1-{len(self.items)}) or 'q': {RESET}").strip()
                if choice.lower() == 'q':
                    print("\033[2J\033[H", end="", flush=True)
                    return None
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(self.items):
                    print("\033[2J\033[H", end="", flush=True)
                    return self.items[choice_idx]["value"]
            except (ValueError, KeyboardInterrupt):
                print("\033[2J\033[H", end="", flush=True)
                return None


def success_message(message: str):
    print(f"{BRIGHT_GREEN}✓ {message}{RESET}")


def error_message(message: str):
    print(f"{RED}✗ {message}{RESET}")


def warning_message(message: str):
    print(f"{BRIGHT_YELLOW}⚠ {message}{RESET}")
