#!/usr/bin/env python3
"""
Interactive menu system with arrow key navigation and colored output
"""

import sys
import tty
import termios
from typing import List, Tuple, Optional
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.table import Table

class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Bright background colors
    BG_BRIGHT_BLACK = '\033[100m'
    BG_BRIGHT_BLUE = '\033[104m'

class MenuItem:
    """Represents a single menu item"""
    def __init__(self, title: str, description: str = "", value: str = None, icon: str = ""):
        self.title = title
        self.description = description
        self.value = value if value is not None else title
        self.icon = icon

class InteractiveMenu:
    """Interactive menu with arrow key navigation using Rich for smooth display"""
    
    def __init__(self, title: str, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.items: List[MenuItem] = []
        self.current_selection = 0
        self.show_current = False
        self.current_value = None
        self.console = Console()
        self.live = None
        self._should_exit = False
        
    def add_item(self, title: str, description: str = "", value: str = None, icon: str = ""):
        """Add a menu item"""
        self.items.append(MenuItem(title, description, value, icon))
        
    def set_current_selection(self, value: str):
        """Set the current/preferred value"""
        self.current_value = value
        self.show_current = True
        # Find and set the current selection index
        for i, item in enumerate(self.items):
            if item.value == value:
                self.current_selection = i
                break
    
    def _get_key(self) -> str:
        """Get a single keypress from the user"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            
            # Handle arrow keys (escape sequences)
            if ch == '\x1b':  # ESC
                ch += sys.stdin.read(2)
                if ch == '\x1b[A':  # Up arrow
                    return 'UP'
                elif ch == '\x1b[B':  # Down arrow
                    return 'DOWN'
                elif ch == '\x1b[C':  # Right arrow
                    return 'RIGHT'
                elif ch == '\x1b[D':  # Left arrow
                    return 'LEFT'
            
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def _render_menu(self):
        """Render the menu using Rich components"""
        # Create title text
        title_text = Text(self.title, style="bold cyan")
        title_text.append("\n" + "=" * len(self.title), style="cyan")
        
        # Build menu content
        content_lines = []
        
        if self.subtitle:
            content_lines.append(Text(self.subtitle, style="cyan"))
            content_lines.append(Text(""))
        
        # Show current preference if available
        if self.show_current and self.current_value:
            for item in self.items:
                if item.value == self.current_value:
                    current_text = Text(f"Current preference: {item.icon} {item.title}", style="bright_green")
                    content_lines.append(current_text)
                    break
            content_lines.append(Text(""))
        
        # Menu items
        for i, item in enumerate(self.items):
            if i == self.current_selection:
                # Selected item - highlight with background
                selected_text = Text(f"  ▶ {item.icon} {item.title}", style="on bright_blue white")
                content_lines.append(selected_text)
                if item.description:
                    desc_text = Text(f"     {item.description}", style="on bright_blue white")
                    content_lines.append(desc_text)
            else:
                # Normal item
                normal_text = Text(f"  {item.icon} {item.title}", style="white")
                content_lines.append(normal_text)
                if item.description:
                    desc_text = Text(f"    {item.description}", style="bright_black")
                    content_lines.append(desc_text)
            content_lines.append(Text(""))
        
        # Instructions
        instructions = [
            "Instructions:",
            "  ↑/↓  Navigate menu",
            "  Enter  Select option", 
            "  q/Ctrl+C  Cancel"
        ]
        
        for instruction in instructions:
            if instruction.startswith("Instructions:"):
                content_lines.append(Text(instruction, style="bright_yellow"))
            else:
                content_lines.append(Text(instruction, style="cyan"))
        
        # Combine title and content
        full_content = Text("\n").join([title_text] + content_lines)
        
        # Create panel
        panel = Panel(
            Align.left(full_content),
            border_style="cyan",
            padding=(1, 2)
        )
        
        return panel
    
    def show(self) -> Optional[str]:
        """Display the interactive menu and return selected value"""
        if not self.items:
            return None
        
        # Use Rich Live for smooth updates
        with Live(
            self._render_menu(),
            console=self.console,
            refresh_per_second=30,
            transient=False,
            auto_refresh=True
        ) as live:
            self.live = live
            
            while not self._should_exit:
                try:
                    key = self._get_key()
                    
                    if key == 'UP':
                        self.current_selection = (self.current_selection - 1) % len(self.items)
                        live.update(self._render_menu())
                    elif key == 'DOWN':
                        self.current_selection = (self.current_selection + 1) % len(self.items)
                        live.update(self._render_menu())
                    elif key in ('\r', '\n'):  # Enter
                        selected_item = self.items[self.current_selection]
                        self._should_exit = True
                        live.stop()
                        self.console.print(f"✓ Selected: {selected_item.icon} {selected_item.title}", style="bright_green")
                        return selected_item.value
                    elif key.lower() == 'q':
                        self._should_exit = True
                        live.stop()
                        self.console.print("Operation cancelled", style="bright_yellow")
                        return None
                    elif key == '\x03':  # Ctrl+C
                        self._should_exit = True
                        live.stop()
                        self.console.print("Operation cancelled", style="bright_yellow")
                        return None
                        
                except KeyboardInterrupt:
                    self._should_exit = True
                    live.stop()
                    self.console.print("Operation cancelled", style="bright_yellow")
                    return None
                except Exception as e:
                    self._should_exit = True
                    live.stop()
                    self.console.print(f"Error reading input: {e}", style="red")
                    return None
        
        return None

def confirm_dialog(message: str, default: bool = False) -> bool:
    """Show a confirmation dialog"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{message}{Colors.RESET}")
    print(f"{Colors.CYAN}[Y/n]{' (default: Yes)' if default else ' (default: No)'}{Colors.RESET}")
    
    try:
        response = input().strip().lower()
        if not response:
            return default
        return response.startswith('y')
    except KeyboardInterrupt:
        print(f"\n{Colors.BRIGHT_YELLOW}Operation cancelled{Colors.RESET}")
        return False

def info_message(message: str, color: str = Colors.BRIGHT_CYAN):
    """Display an info message with color"""
    print(f"\n{color}{message}{Colors.RESET}")

def success_message(message: str):
    """Display a success message"""
    print(f"{Colors.BRIGHT_GREEN}✓ {message}{Colors.RESET}")

def error_message(message: str):
    """Display an error message"""
    print(f"{Colors.BRIGHT_RED}✗ {message}{Colors.RESET}")

def warning_message(message: str):
    """Display a warning message"""
    print(f"{Colors.BRIGHT_YELLOW}⚠ {message}{Colors.RESET}")
