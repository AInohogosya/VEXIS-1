#!/usr/bin/env python3
"""
Example usage of the Yellow Highlight Navigation System
Demonstrates basic menu functionality with arrow key navigation
"""

from yellow_highlight_navigation import get_yellow_menu

def main():
    print("🟡 Yellow Highlight Navigation Demo")
    print("=" * 40)
    print("Use arrow keys ↑↓ to navigate, Enter to select")
    print("Press 'q' to quit")
    print("=" * 40)
    print()

    # Create a menu
    menu = get_yellow_menu(
        "🚀 Select Your AI Model",
        "Choose the AI model that best fits your needs:"
    )

    # Add some example options
    menu.add_item(
        "GPT-4",
        "Most capable model • Advanced reasoning • 128K context",
        "gpt-4",
        "🧠"
    )

    menu.add_item(
        "GPT-3.5 Turbo",
        "Fast and efficient • Good for most tasks • 16K context",
        "gpt-3.5-turbo",
        "⚡"
    )

    menu.add_item(
        "Claude 3 Opus",
        "Excellent writing • Strong analysis • 200K context",
        "claude-3-opus",
        "✍️"
    )

    menu.add_item(
        "Gemini 1.5 Pro",
        "Google's latest • Multimodal • 1M context",
        "gemini-1.5-pro",
        "🌟"
    )

    # Show the menu and get selection
    selected_model = menu.show()

    if selected_model:
        print(f"\n✅ You selected: {selected_model}")
        print("🎉 The yellow highlight navigation system is working!")
    else:
        print("\n👋 Selection cancelled")

if __name__ == "__main__":
    main()
