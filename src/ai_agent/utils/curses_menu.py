#!/usr/bin/env python3
"""
Curses-based Arrow Key Menu System for VEXIS-1.2
Proper arrow key navigation optimized for Gemini model selection
Works in any terminal that supports curses
"""

import curses
import os
from typing import Optional, List, Dict, Any, Callable

# Color pairs
COLOR_TITLE = 1
COLOR_HIGHLIGHT = 2
COLOR_NORMAL = 3
COLOR_FOOTER = 4
COLOR_SUCCESS = 5
COLOR_WARNING = 6


class CursesMenu:
    """Curses-based interactive menu with arrow key navigation"""
    
    def __init__(self, title: str, description: str = ""):
        self.title = title
        self.description = description
        self.items: List[Dict[str, Any]] = []
        self.current_index = 0
        self.selected_index = None
    
    def add_item(self, display_name: str, description: str, value: Any, icon: str = "📋", 
                 installed: bool = False, category: str = ""):
        """Add an item to the menu"""
        self.items.append({
            "display_name": display_name,
            "description": description,
            "value": value,
            "icon": icon,
            "installed": installed,
            "category": category
        })
    
    def add_separator(self, text: str = ""):
        """Add a separator line"""
        self.items.append({
            "separator": True,
            "text": text
        })
    
    def run(self, stdscr) -> Optional[Any]:
        """Run the menu and return selected value"""
        # Configure curses
        curses.curs_set(0)  # Hide cursor
        stdscr.clear()
        
        # Initialize colors
        if curses.has_colors():
            curses.start_color()
            curses.init_pair(COLOR_TITLE, curses.COLOR_CYAN, curses.COLOR_BLACK)
            curses.init_pair(COLOR_HIGHLIGHT, curses.COLOR_BLACK, curses.COLOR_YELLOW)
            curses.init_pair(COLOR_NORMAL, curses.COLOR_WHITE, curses.COLOR_BLACK)
            curses.init_pair(COLOR_FOOTER, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            curses.init_pair(COLOR_SUCCESS, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(COLOR_WARNING, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        
        # Calculate layout
        max_y, max_x = stdscr.getmaxyx()
        
        while True:
            stdscr.clear()
            
            # Title
            if len(self.title) < max_x:
                title_attr = curses.A_BOLD | curses.color_pair(COLOR_TITLE) if curses.has_colors() else curses.A_BOLD
                stdscr.addstr(0, 0, self.title, title_attr)
            
            # Separator
            separator = "=" * min(50, max_x - 1)
            sep_attr = curses.color_pair(COLOR_TITLE) if curses.has_colors() else curses.A_DIM
            stdscr.addstr(1, 0, separator, sep_attr)
            
            # Description
            if self.description and len(self.description) < max_x:
                desc_attr = curses.color_pair(COLOR_NORMAL) if curses.has_colors() else curses.A_NORMAL
                stdscr.addstr(2, 0, self.description, desc_attr)
            
            # Instructions
            if max_y > 4:
                instructions = "💡 Use ↑↓ arrows to navigate • Enter to select • Q to quit"
                if any(item.get("installed", False) for item in self.items):
                    instructions += " • I for installed only"
                inst_attr = curses.color_pair(COLOR_FOOTER) if curses.has_colors() else curses.A_DIM
                stdscr.addstr(4, 0, instructions, inst_attr)
            
            # Menu items
            start_y = 6
            display_items = []
            
            for i, item in enumerate(self.items):
                if item.get("separator", False):
                    continue  # Skip separators in navigation
                display_items.append((i, item))
            
            for display_i, (original_i, item) in enumerate(display_items):
                y = start_y + (display_i * 3)
                if y >= max_y - 3:
                    break
                
                if original_i == self.current_index:
                    # Highlighted selection
                    status_icon = "✓" if item.get("installed", False) else "○"
                    line1 = f"  ▶ {status_icon} {item['icon']} {item['display_name']}"
                    line2 = f"     {item['description']}"
                    
                    if len(line1) < max_x:
                        attr1 = curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BOLD if curses.has_colors() else curses.A_REVERSE
                        stdscr.addstr(y, 0, line1, attr1)
                    if len(line2) < max_x:
                        attr2 = curses.color_pair(COLOR_HIGHLIGHT) if curses.has_colors() else curses.A_REVERSE
                        stdscr.addstr(y + 1, 0, line2, attr2)
                else:
                    # Normal option
                    if item.get("installed", False):
                        status_icon = "✓"
                        status_attr = curses.color_pair(COLOR_SUCCESS) if curses.has_colors() else curses.A_NORMAL
                    else:
                        status_icon = "○"
                        status_attr = curses.color_pair(COLOR_NORMAL) if curses.has_colors() else curses.A_NORMAL
                    
                    line1 = f"  {status_icon} {item['icon']} {item['display_name']}"
                    line2 = f"     {item['description']}"
                    
                    if len(line1) < max_x:
                        stdscr.addstr(y, 0, line1, status_attr)
                    if len(line2) < max_x:
                        desc_attr = curses.color_pair(COLOR_NORMAL) if curses.has_colors() else curses.A_DIM
                        stdscr.addstr(y + 1, 0, line2, desc_attr)
            
            # Footer with model count
            if max_y > start_y + len(display_items) * 3 + 2:
                installed_count = sum(1 for item in self.items if item.get("installed", False))
                total_count = len([item for item in self.items if not item.get("separator", False)])
                footer_text = f"Models: {installed_count}/{total_count} installed"
                footer_attr = curses.color_pair(COLOR_FOOTER) if curses.has_colors() else curses.A_DIM
                stdscr.addstr(max_y - 2, 0, footer_text, footer_attr)
            
            stdscr.refresh()
            
            # Handle input
            key = stdscr.getch()
            
            if key == curses.KEY_UP:
                self._move_up()
            elif key == curses.KEY_DOWN:
                self._move_down()
            elif key == ord('q') or key == ord('Q'):
                return None
            elif key == ord('\n') or key == curses.KEY_ENTER:
                if 0 <= self.current_index < len(self.items):
                    item = self.items[self.current_index]
                    if not item.get("separator", False):
                        self.selected_index = self.current_index
                        return item["value"]
            elif key == ord('i') or key == ord('I'):
                # Filter to installed only
                self._filter_installed()
            
            # Page up/down for longer lists
            elif key == curses.KEY_PPAGE:
                self._move_page_up()
            elif key == curses.KEY_NPAGE:
                self._move_page_down()
            
            # Home/End
            elif key == curses.KEY_HOME:
                self.current_index = 0
            elif key == curses.KEY_END:
                self.current_index = len(self.items) - 1
    
    def _move_up(self):
        """Move selection up"""
        # Find previous non-separator item
        new_index = self.current_index - 1
        while new_index >= 0:
            if not self.items[new_index].get("separator", False):
                self.current_index = new_index
                break
            new_index -= 1
    
    def _move_down(self):
        """Move selection down"""
        # Find next non-separator item
        new_index = self.current_index + 1
        while new_index < len(self.items):
            if not self.items[new_index].get("separator", False):
                self.current_index = new_index
                break
            new_index += 1
    
    def _move_page_up(self):
        """Move up by page"""
        self.current_index = max(0, self.current_index - 5)
        self._move_to_valid_item()
    
    def _move_page_down(self):
        """Move down by page"""
        self.current_index = min(len(self.items) - 1, self.current_index + 5)
        self._move_to_valid_item()
    
    def _move_to_valid_item(self):
        """Move to nearest valid (non-separator) item"""
        while 0 <= self.current_index < len(self.items):
            if not self.items[self.current_index].get("separator", False):
                break
            self.current_index += 1
    
    def _filter_installed(self):
        """Filter to show only installed models"""
        installed_indices = [i for i, item in enumerate(self.items) 
                           if item.get("installed", False) and not item.get("separator", False)]
        if installed_indices:
            # Find the next installed model after current position
            for idx in installed_indices:
                if idx > self.current_index:
                    self.current_index = idx
                    return
            # If no installed model after current, go to first installed
            self.current_index = installed_indices[0]


def get_curses_menu(title: str, description: str = "") -> CursesMenu:
    """Get a new curses menu instance"""
    return CursesMenu(title, description)


def show_model_selection_menu(models: List[Dict[str, Any]], title: str = "Select Model") -> Optional[str]:
    """Show a model selection menu"""
    def menu_runner(stdscr):
        menu = CursesMenu(title, "Choose your preferred model:")
        
        # Group models by category
        categories = {}
        for model in models:
            category = model.get("category", "other")
            if category not in categories:
                categories[category] = []
            categories[category].append(model)
        
        # Add models in category order
        category_order = ["lightweight", "balanced", "performance", "cloud", "other"]
        
        for cat in category_order:
            if cat in categories:
                if cat != "other":
                    menu.add_separator(f"--- {cat.title()} Models ---")
                for model in categories[cat]:
                    menu.add_item(
                        model["display_name"],
                        model["description"],
                        model["name"],
                        model.get("icon", "📋"),
                        model.get("installed", False),
                        model.get("category", "")
                    )
        
        return menu.run(stdscr)
    
    try:
        return curses.wrapper(menu_runner)
    except Exception as e:
        print(f"Error showing menu: {e}")
        return None


if __name__ == "__main__":
    # Test the menu system
    test_models = [
        {"name": "gemma2:2b", "display_name": "Gemma 2 2B", "description": "2B parameters • Efficient", "icon": "⚡", "installed": True, "category": "lightweight"},
        {"name": "gemma3:4b", "display_name": "Gemma 3 4B", "description": "4B parameters • Multimodal", "icon": "🪶", "installed": False, "category": "balanced"},
        {"name": "gemini-3-flash-preview", "display_name": "Gemini 3 Flash", "description": "Speed optimized • Cloud", "icon": "☁️", "installed": False, "category": "cloud"},
    ]
    
    selected = show_model_selection_menu(test_models)
    if selected:
        print(f"Selected: {selected}")
    else:
        print("No selection made")
