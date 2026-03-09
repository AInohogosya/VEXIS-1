# 🟡 Yellow Highlight Navigation System

## Overview
A standalone, reusable menu system with consistent yellow highlighting for selection interfaces. Perfect for CLI applications that need interactive model or option selection with arrow key navigation.

## Features
- 🟡 **Consistent Yellow Highlighting**: All selections use the same yellow color scheme
- 🎯 **Reproducible Configuration**: Centralized settings for cross-platform consistency
- 🧹 **Clean Display**: In-place updates without creating confusing logs
- 🎮 **Multiple Navigation**: Arrow keys, number keys, and Enter support
- 📱 **Cross-Platform**: Works on Windows, Linux, and macOS
- 📦 **Standalone**: No external dependencies, pure Python standard library

## Quick Start

```python
from yellow_highlight_navigation import get_yellow_menu

# Create a menu
menu = get_yellow_menu("Select Option", "Choose your preference:")

# Add items
menu.add_item("Option 1", "Description of option 1", "value1", "📋")
menu.add_item("Option 2", "Description of option 2", "value2", "🔧")

# Show menu and get selection
selected = menu.show()
print(f"Selected: {selected}")
```

## Files

### Core Components
- **`__init__.py`**: Package initialization and exports
- **`main.py`**: Main entry point and convenience functions
- **`config.py`**: Reproducible configuration settings
- **`clean_interactive_menu.py`**: Core menu implementation with yellow highlighting
- **`clean_hierarchical_selector.py`**: Hierarchical model selection
- **`fallback_interactive_menu.py`**: Fallback for terminals without cursor positioning

## Usage Examples

### Basic Menu
```python
from yellow_highlight_navigation import get_yellow_menu

menu = get_yellow_menu("Select Option", "Choose your preference:")
menu.add_item("Option 1", "Description", "value1", "📋")
selected = menu.show()
```

### Hierarchical Selection
```python
from yellow_highlight_navigation import get_yellow_selector

selector = get_yellow_selector()
selected_model = selector.interactive_model_selection()
```

### Provider Selection
```python
from yellow_highlight_navigation.main import create_provider_menu

menu = create_provider_menu()
menu.add_item("Local", "Run locally", "local", "🏠")
menu.add_item("Cloud", "Run in cloud", "cloud", "☁️")
selected_provider = menu.show()
```

## Installation

Just copy the `yellow-highlight-navigation` folder into your project and import:

```python
from yellow_highlight_navigation import get_yellow_menu
```

## Navigation

- **↑/↓ Arrow Keys**: Navigate up/down
- **Enter**: Select highlighted option
- **Number Keys**: Quick select (1-9)
- **q/Q**: Quit/cancel selection

## Dependencies

None! This package uses only Python's standard library:
- `sys`
- `os` 
- `typing`

## Compatibility

- Python 3.7+
- All major operating systems
- Any terminal that supports ANSI escape codes

## Version
- **Version**: 1.0.0
- **License**: MIT (inferred from original project)

## Demo

Run the demo to see it in action:

```bash
cd yellow-highlight-navigation
python main.py
```

Or from your code:

```python
from yellow_highlight_navigation import show_yellow_selection_demo
show_yellow_selection_demo()
```
