"""
Clean Hierarchical Model Selector
Updates existing display without creating new content below
"""

from typing import Optional, Dict, Any
try:
    from .clean_interactive_menu import CleanInteractiveMenu, Colors, success_message, error_message, warning_message
except ImportError:
    # Fallback for direct execution
    from clean_interactive_menu import CleanInteractiveMenu, Colors, success_message, error_message, warning_message


class CleanHierarchicalSelector:
    """Clean hierarchical selector that updates display in-place"""
    
    MODEL_FAMILIES = {
        "microsoft": {
            "name": "Microsoft",
            "description": "Microsoft's Phi family models",
            "icon": "🔷",
            "subfamilies": {
                "phi3": {
                    "name": "Phi-3",
                    "description": "Lightweight state-of-the-art open models",
                    "icon": "🧠",
                    "models": {
                        "phi3:mini": {"name": "Phi-3 Mini", "desc": "3.8B parameters • Lightweight • 4K context", "icon": "⚡"},
                        "phi3:medium": {"name": "Phi-3 Medium", "desc": "14B parameters • Strong reasoning • 4K context", "icon": "🧠"},
                        "phi3:medium-128k": {"name": "Phi-3 Medium 128K", "desc": "14B parameters • Strong reasoning • 128K context", "icon": "🌐"},
                    }
                },
                "phi4": {
                    "name": "Phi-4",
                    "description": "Latest generation Phi models",
                    "icon": "🚀",
                    "models": {
                        "phi4:latest": {"name": "Phi-4 Latest", "desc": "14B parameters • State-of-the-art • 16K context", "icon": "⭐"},
                    }
                }
            }
        },
        "google": {
            "name": "Google",
            "description": "Google's Gemma family models",
            "icon": "🔍",
            "subfamilies": {
                "gemma3": {
                    "name": "Gemma 3",
                    "description": "Latest generation Gemma models",
                    "icon": "💎",
                    "models": {
                        "gemma3:1b": {"name": "Gemma 3 1B", "desc": "1B parameters • Fast and lightweight • 32k context", "icon": "⚡"},
                        "gemma3:4b": {"name": "Gemma 3 4B Vision", "desc": "4B parameters • Vision capable • 128k context", "icon": "👁"},
                    }
                }
            }
        },
        "alibaba": {
            "name": "Alibaba",
            "description": "Alibaba's Qwen family models",
            "icon": "🐲",
            "subfamilies": {
                "qwen3": {
                    "name": "Qwen 3",
                    "description": "Latest generation Qwen models",
                    "icon": "🌏",
                    "models": {
                        "qwen3:1.7b": {"name": "Qwen 3 1.7B", "desc": "1.7B parameters • Efficient multilingual • 40K context", "icon": "⚡"},
                        "qwen3:8b": {"name": "Qwen 3 8B", "desc": "8B parameters • Strong performance • 40K context", "icon": "🧠"},
                    }
                }
            }
        }
    }
    
    def __init__(self):
        """Initialize clean selector"""
        pass
    
    def display_model_families(self) -> Optional[str]:
        """Display model families with clean updates"""
        menu = CleanInteractiveMenu(
            "🤖 Select Model Family",
            "Choose the AI model family you want to explore:"
        )
        
        for key, data in self.MODEL_FAMILIES.items():
            menu.add_item(
                data["name"],
                data["description"],
                key,
                data["icon"]
            )
        
        return menu.show()
    
    def display_subfamilies(self, family_key: str) -> Optional[str]:
        """Display subfamilies with clean updates"""
        if family_key not in self.MODEL_FAMILIES:
            return None
        
        family_data = self.MODEL_FAMILIES[family_key]
        menu = CleanInteractiveMenu(
            f"🔷 {family_data['name']} Subfamilies",
            f"Choose the {family_data['name']} model series:"
        )
        
        for key, data in family_data["subfamilies"].items():
            menu.add_item(
                data["name"],
                data["description"],
                key,
                data["icon"]
            )
        
        return menu.show()
    
    def display_models_in_subfamily(self, family_key: str, subfamily_key: str) -> Optional[str]:
        """Display models with clean updates"""
        if (family_key not in self.MODEL_FAMILIES or 
            subfamily_key not in self.MODEL_FAMILIES[family_key]["subfamilies"]):
            return None
        
        subfamily_data = self.MODEL_FAMILIES[family_key]["subfamilies"][subfamily_key]
        menu = CleanInteractiveMenu(
            f"🔧 {subfamily_data['name']} Models",
            f"Select your preferred {subfamily_data['name']} model:"
        )
        
        for key, data in subfamily_data["models"].items():
            menu.add_item(
                data["name"],
                data["desc"],
                key,
                data["icon"]
            )
        
        return menu.show()
    
    def show_final_selection(self, family_key: str, subfamily_key: str, model_key: str):
        """Display final selection"""
        family_data = self.MODEL_FAMILIES[family_key]
        subfamily_data = family_data["subfamilies"][subfamily_key]
        model_data = subfamily_data["models"][model_key]
        
        # Clear screen for final display
        print("\033[2J\033[H", end="", flush=True)
        
        print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}🎉 SELECTION COMPLETE{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}Family:{Colors.RESET}    {family_data['icon']} {family_data['name']}")
        print(f"{Colors.BRIGHT_CYAN}Subfamily:{Colors.RESET} {subfamily_data['icon']} {subfamily_data['name']}")
        print(f"{Colors.BRIGHT_CYAN}Model:{Colors.RESET}    {model_data['icon']} {model_data['name']}")
        print(f"{Colors.BRIGHT_CYAN}Details:{Colors.RESET}   {model_data['desc']}")
        print(f"{Colors.BRIGHT_CYAN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BRIGHT_GREEN}✨ Perfect choice for your AI needs!{Colors.RESET}")
        print()
    
    def interactive_model_selection(self) -> Optional[str]:
        """Clean hierarchical model selection with in-place updates"""
        # Clear screen once at start
        print("\033[2J\033[H", end="", flush=True)
        
        print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}🚀 Clean Ollama Model Selection{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}💡 Clean display updates • No confusing logs below{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{'=' * 60}{Colors.RESET}")
        print()
        
        # Step 1: Select family
        family_key = self.display_model_families()
        if not family_key:
            print("👋 Model selection cancelled")
            return None
        
        # Step 2: Select subfamily
        subfamily_key = self.display_subfamilies(family_key)
        if not subfamily_key:
            print("👋 Model selection cancelled")
            return None
        
        # Step 3: Select specific model
        model_key = self.display_models_in_subfamily(family_key, subfamily_key)
        if not model_key:
            print("👋 Model selection cancelled")
            return None
        
        # Show final selection
        self.show_final_selection(family_key, subfamily_key, model_key)
        return model_key


def get_clean_selector() -> CleanHierarchicalSelector:
    """Get clean hierarchical selector instance"""
    return CleanHierarchicalSelector()
