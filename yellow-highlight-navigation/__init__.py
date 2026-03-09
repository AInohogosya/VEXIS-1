"""
Yellow Selection System - Unified Menu Technology
Clean, consistent yellow highlighting for all selection interfaces
"""

from .clean_interactive_menu import CleanInteractiveMenu, Colors, success_message, error_message, warning_message
from .clean_hierarchical_selector import get_clean_selector, CleanHierarchicalSelector
from .fallback_interactive_menu import FallbackInteractiveMenu
from .main import get_yellow_menu, get_yellow_selector, show_yellow_selection_demo

__all__ = [
    'CleanInteractiveMenu',
    'Colors', 
    'success_message',
    'error_message', 
    'warning_message',
    'get_clean_selector',
    'CleanHierarchicalSelector',
    'get_yellow_menu',
    'get_yellow_selector',
    'show_yellow_selection_demo',
    'FallbackInteractiveMenu'
]

__version__ = "1.0.0"
__description__ = "Unified yellow selection system with clean interface"
