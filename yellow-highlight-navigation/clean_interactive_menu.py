"""
Clean Interactive Menu - No Log Creation
Updates existing display without creating new content below
"""

import sys
import os
import tty
import termios
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


class CleanInteractiveMenu:
    """Clean menu that updates display without creating new content"""
    
    def __init__(self, title: str, description: str):
        self.title = title
        self.description = description
        self.items = []
        self.current_index = 0
        self.displayed_once = False
        
    def add_item(self, display_name: str, description: str, value: Any, icon: str = "📋"):
        self.items.append({
            "display_name": display_name,
            "description": description,
            "value": value,
            "icon": icon
        })
    
    def clear_screen(self):
        """Clear screen only once"""
        if not self.displayed_once:
            print("\033[2J\033[H", end="", flush=True)
            self.displayed_once = True
    
    def display_header(self):
        """Display header only once"""
        if not self.displayed_once:
            print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}🔧 {self.title}{Colors.RESET}")
            print(f"{Colors.BRIGHT_CYAN}{'─' * 50}{Colors.RESET}")
            print(f"{Colors.CYAN}📝 {self.description}{Colors.RESET}")
            print()
            print(f"{Colors.BOLD}📋 Available Options:{Colors.RESET}")
            print()
    
    def display_footer(self):
        """Display footer only once"""
        if not self.displayed_once:
            print(f"{Colors.BRIGHT_CYAN}{'─' * 50}{Colors.RESET}")
            print(f"{Colors.BRIGHT_YELLOW}💡 Use ↑↓ arrows • 1-9 numbers • Enter to select • 'q' to quit{Colors.RESET}")
            # Add extra lines to prevent footer overlap
            print()
            print()
    
    def update_display(self):
        """Update only the menu items, no new content"""
        # Calculate starting line for menu items (after header)
        header_lines = 6  # Title, separator, description, empty, "Available Options", empty
        
        for i, item in enumerate(self.items):
            # Calculate actual line positions
            item_line = header_lines + (i * 3)  # 3 lines per item (item, description, empty)
            desc_line = item_line + 1
            
            # Update item line
            print(f"\033[{item_line};0H", end="", flush=True)  # Move to item line
            print(f"\033[K", end="", flush=True)  # Clear line
            
            if i == self.current_index:
                # Highlighted selection with proper yellow background
                print(f"{Colors.BG_YELLOW}{Colors.BLACK}  ▶ [{i+1}] {item['icon']} {item['display_name']}{Colors.RESET}", end="", flush=True)
            else:
                # Normal option
                print(f"  [{i+1}] {item['icon']} {item['display_name']}", end="", flush=True)
            
            # Update description line
            print(f"\033[{desc_line};0H", end="", flush=True)  # Move to description line
            print(f"\033[K", end="", flush=True)  # Clear description line
            
            if i == self.current_index:
                print(f"{Colors.BG_YELLOW}{Colors.BLACK}       {item['description']}{Colors.RESET}", end="", flush=True)
            else:
                print(f"       {item['description']}", end="", flush=True)
    
    def get_key(self) -> str:
        """Get key press with proper terminal raw mode"""
        try:
            return self._get_key_raw_mode()
        except Exception:
            # Fallback to simple input
            return self._fallback_input()
    
    def _get_key_raw_mode(self) -> str:
        """Get key using raw terminal mode for proper arrow key detection"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            
            # Handle arrow keys (escape sequences)
            if ch == '\x1b':  # ESC
                # Read the next characters to get the full sequence
                ch += sys.stdin.read(2)
                if ch == '\x1b[A':  # Up arrow
                    return '\x1b[A'
                elif ch == '\x1b[B':  # Down arrow
                    return '\x1b[B'
                elif ch == '\x1b[C':  # Right arrow
                    return '\x1b[C'
                elif ch == '\x1b[D':  # Left arrow
                    return '\x1b[D'
                else:
                    return ch  # Return other escape sequences as-is
            
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
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
    
    def show(self) -> Optional[Any]:
        """Show clean interactive menu with improved display handling"""
        try:
            # Clear screen once at start
            print("\033[2J\033[H", end="", flush=True)
            self.displayed_once = True
            self.display_header()
            
            while True:
                # Move cursor back to the start of menu items
                print("\033[6;0H", end="", flush=True)  # Move to line 6 (after header)
                
                # Clear and redraw the entire menu area
                self._display_menu_items()
                self.display_footer()
                
                # Get key input
                key = self.get_key()
                
                # Skip empty keys (no input)
                if not key or key == '':
                    continue
                
                # Handle input
                if isinstance(key, str) and key.lower() == 'q':
                    print("\033[2J\033[H", end="", flush=True)
                    return None
                elif key in ['\r', '\n']:  # Enter key
                    selected_item = self.items[self.current_index]
                    print("\033[2J\033[H", end="", flush=True)
                    return selected_item["value"]
                elif key == '\x1b[A':  # Up arrow
                    self.current_index = max(0, self.current_index - 1)
                elif key == '\x1b[B':  # Down arrow
                    self.current_index = min(len(self.items) - 1, self.current_index + 1)
                elif isinstance(key, str) and key.isdigit():  # Number input
                    try:
                        num = int(key)
                        if 1 <= num <= len(self.items):
                            self.current_index = num - 1
                    except:
                        pass
                    
        except KeyboardInterrupt:
            print("\033[2J\033[H", end="", flush=True)
            return None
        except Exception:
            return self.fallback_selection()
    
    def _display_menu_items(self):
        """Display all menu items"""
        # Clear the menu area and redraw
        for i, item in enumerate(self.items):
            if i == self.current_index:
                # Highlighted selection with yellow background
                print(f"{Colors.BG_YELLOW}{Colors.BLACK}  ▶ [{i+1}] {item['icon']} {item['display_name']}{Colors.RESET}")
                print(f"{Colors.BG_YELLOW}{Colors.BLACK}       {item['description']}{Colors.RESET}")
            else:
                # Normal option
                print(f"  [{i+1}] {item['icon']} {item['display_name']}")
                print(f"       {item['description']}")
            print()
    
    def fallback_selection(self) -> Optional[Any]:
        """Fallback to numbered selection with clean display"""
        print(f"\n{Colors.RED}⚠️  Using simple selection as fallback{Colors.RESET}")
        print()
        
        for i, item in enumerate(self.items, 1):
            print(f"{i}. {item['icon']} {item['display_name']}")
            print(f"   {item['description']}")
            print()
        
        while True:
            try:
                choice = input(f"{Colors.YELLOW}Select (1-{len(self.items)}) or 'q': {Colors.RESET}").strip()
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
    print(f"{Colors.BRIGHT_GREEN}✓ {message}{Colors.RESET}")


def error_message(message: str):
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def warning_message(message: str):
    print(f"{Colors.BRIGHT_YELLOW}⚠ {message}{Colors.RESET}")


def info_message(message: str):
    print(f"{Colors.BRIGHT_CYAN}ℹ {message}{Colors.RESET}")
