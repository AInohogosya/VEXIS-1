"""
Yellow Selection System - Main Entry Point
Central access point for all yellow selection functionality
"""

try:
    from .clean_interactive_menu import CleanInteractiveMenu, Colors, success_message, error_message, warning_message
    from .clean_hierarchical_selector import get_clean_selector, CleanHierarchicalSelector
    from .fallback_interactive_menu import FallbackInteractiveMenu
except ImportError:
    # Fallback for direct execution
    from clean_interactive_menu import CleanInteractiveMenu, Colors, success_message, error_message, warning_message
    from clean_hierarchical_selector import get_clean_selector, CleanHierarchicalSelector
    from fallback_interactive_menu import FallbackInteractiveMenu


def get_yellow_menu(title: str, description: str, use_fallback: bool = False) -> CleanInteractiveMenu:
    """Get a yellow-highlighted interactive menu"""
    if use_fallback:
        return FallbackInteractiveMenu(title, description)
    else:
        return CleanInteractiveMenu(title, description)


def get_yellow_selector(use_fallback: bool = False) -> CleanHierarchicalSelector:
    """Get the yellow hierarchical selector"""
    return get_clean_selector()


def show_yellow_selection_demo():
    """Demonstrate the yellow selection system"""
    print("🟡 Yellow Selection System Demo")
    print("=" * 50)
    print("Clean, consistent yellow highlighting")
    print("=" * 50)
    print()
    
    selector = get_yellow_selector()
    selected_model = selector.interactive_model_selection()
    
    if selected_model:
        print(f"\n✨ Demo completed! Selected: {selected_model}")
    else:
        print("\n👋 Demo cancelled")


# Convenience functions for common use cases
def create_provider_menu() -> CleanInteractiveMenu:
    """Create a provider selection menu with yellow highlighting"""
    return get_yellow_menu(
        "Select AI Provider",
        "Choose how you want to run AI models:"
    )


def create_model_menu() -> CleanInteractiveMenu:
    """Create a model selection menu with yellow highlighting"""
    return get_yellow_menu(
        "Select Model", 
        "Choose your preferred AI model:"
    )


# Export main components
__all__ = [
    'get_yellow_menu',
    'get_yellow_selector', 
    'show_yellow_selection_demo',
    'create_provider_menu',
    'create_model_menu'
]
