#!/usr/bin/env python3
"""
Configuration Manager
Handles model provider configuration and selection
"""

import subprocess
import getpass
from typing import Optional, Tuple

from .settings_manager import get_settings_manager
import sys
import os
# Add the yellow-highlight-navigation directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(project_root, 'yellow-highlight-navigation'))
from clean_interactive_menu import CleanInteractiveMenu, Colors, success_message, error_message, warning_message, info_message
from .ollama_manager import OllamaManager
from .model_definitions import get_cloud_gemini_models, get_local_gemini_models, get_gemini_model_info
from .ollama_error_handler import handle_ollama_error


class ConfigManager:
    """Manages AI provider configuration"""
    
    def __init__(self):
        self.settings_manager = get_settings_manager()
        self.ollama_manager = OllamaManager()
    
    def check_ollama_login(self) -> bool:
        """Check if Ollama is logged in and prompt for sign-in if needed"""
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                error_message("Ollama is not installed or not in PATH")
                print(f"{Colors.BRIGHT_CYAN}Please install Ollama first: https://ollama.com/{Colors.RESET}")
                return False
            
            result = subprocess.run(["ollama", "whoami"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0 or "not signed in" in result.stderr.lower():
                warning_message("Ollama is available but you are not signed in.")
                print(f"{Colors.CYAN}Running 'ollama signin' to sign in...{Colors.RESET}")
                
                signin_result = subprocess.run(["ollama", "signin"], timeout=120)
                if signin_result.returncode == 0:
                    success_message("Ollama sign-in completed")
                    return True
                else:
                    error_message("Ollama sign-in failed")
                    handle_ollama_error(signin_result.stderr or "Sign-in failed", 
                                     {"operation": "signin"}, display_to_user=True)
                    return False
            else:
                success_message("Ollama is signed in")
                return True
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            error_message(f"Error checking Ollama: {e}")
            return False
    
    def prompt_google_api_key(self) -> Optional[Tuple[str, bool]]:
        """Prompt user for Google API key"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Google API Key Setup{Colors.RESET}")
        print(f"{Colors.CYAN}{'-' * 25}{Colors.RESET}")
        print(f"{Colors.WHITE}To use Google's official Gemini API, you need an API key.{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}You can get one from: https://aistudio.google.com/app/apikey{Colors.RESET}")
        print()
        
        try:
            api_key = getpass.getpass(f"{Colors.YELLOW}Enter your Google API key (or press Enter to cancel):{Colors.RESET} ")
            if not api_key.strip():
                warning_message("No API key provided. Skipping Google API setup.")
                return None
            
            if len(api_key) < 20:
                error_message("API key seems too short. Please check your key.")
                return self.prompt_google_api_key()
            
            save_key = input(f"{Colors.CYAN}Save this API key for future use? (y/n):{Colors.RESET} ").lower().strip()
            should_save = save_key.startswith('y')
            
            return api_key, should_save
            
        except (KeyboardInterrupt, Exception) as e:
            error_message(f"Error reading input: {e}")
            return None
    
    def select_google_model(self) -> Optional[str]:
        """Select Google model using yellow highlight menu"""
        current_model = self.settings_manager.get_google_model()
        
        menu = CleanInteractiveMenu("🚀 Select Gemini Model", "Choose your preferred Gemini model:")
        
        cloud_models = get_cloud_gemini_models()
        for model_name in cloud_models:
            model_info = get_gemini_model_info(model_name)
            menu.add_item(
                model_info.get('name', model_name),
                model_info.get('desc', ''),
                model_name,
                model_info.get('icon', '📋')
            )
        
        selected_model = menu.show()
        
        if selected_model is None:
            return current_model
        
        self.settings_manager.set_google_model(selected_model)
        return selected_model
    
    def configure_google_provider(self) -> Optional[Tuple[str, str]]:
        """Configure Google provider with API key and model selection"""
        if not self.settings_manager.has_google_api_key():
            result = self.prompt_google_api_key()
            if result is None:
                return None
            
            api_key, should_save = result
            self.settings_manager.set_google_api_key(api_key, should_save)
        
        model = self.select_google_model()
        if model is None:
            model = self.settings_manager.get_google_model()
        
        self.settings_manager.set_preferred_provider("google")
        return "google", model
    
    def configure_ollama_provider(self) -> Optional[str]:
        """Configure Ollama provider with model selection"""
        if not self.ollama_manager.check_ollama_available():
            error_message("Ollama is not available")
            print(f"{Colors.BRIGHT_CYAN}Please install Ollama first: https://ollama.com/{Colors.RESET}")
            return None
        
        if not self.check_ollama_login():
            warning_message("Ollama login failed. You can still use local models.")
        
        menu = CleanInteractiveMenu("🦊 Select Ollama Model", "Choose your preferred Gemini model:")
        
        gemini_models = self.ollama_manager.list_gemini_models(installed_only=False)
        
        # Directly add the two models without categorization
        for model in gemini_models:
            status_icon = "✓" if model.get('installed', False) else "○"
            menu.add_item(
                f"{status_icon} {model['display_name']}",
                model['description'],
                model['name'],
                model.get('icon', '📋')
            )
        
        selected_model = menu.show()
        
        if selected_model is None:
            return None
        
        if not self.ollama_manager.is_model_installed(selected_model):
            info_message(f"Installing {selected_model}...")
            install_result = self.ollama_manager.install_model(selected_model)
            if not install_result['success']:
                handle_ollama_error(install_result.get('error', 'Installation failed'), 
                                 {'model_name': selected_model, 'operation': 'pull_model'}, 
                                 display_to_user=True)
                return None
        
        self.settings_manager.set_preferred_provider("ollama")
        return "ollama"
    
    def select_provider(self) -> Optional[str]:
        """Main configuration screen for model provider selection"""
        current_provider = self.settings_manager.get_preferred_provider()
        
        print(f"\033[2J\033[H", end="")
        
        print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'═' * 50}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}     VEXIS-1.2 AI Agent Configuration{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'═' * 50}{Colors.RESET}\n")
        
        menu = CleanInteractiveMenu("Select AI Provider", "Choose how you want to run AI models:")
        
        menu.add_item("Ollama (Local)", "Run Gemini models locally via Ollama • Privacy-focused", "ollama", "🦊")
        menu.add_item("Google Official API", "Use Google's cloud Gemini models • Requires API key", "google", "🌐")
        
        if current_provider:
            for i, item in enumerate(menu.items):
                if item["value"] == current_provider:
                    menu.current_index = i
                    break
        
        selected_provider = menu.show()
        
        if selected_provider is None:
            return current_provider
        
        if selected_provider == "ollama":
            result = self.configure_ollama_provider()
            return result if result is not None else self.select_provider()
        elif selected_provider == "google":
            result = self.configure_google_provider()
            return result[0] if result is not None else self.select_provider()
        
        return current_provider
    
    def show_config_summary(self, provider: str, model: str = None):
        """Display configuration summary"""
        print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{'─' * 50}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}✓ Configuration Complete{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'─' * 50}{Colors.RESET}")
        
        if provider == "ollama":
            print(f"{Colors.WHITE}  Provider: {Colors.BRIGHT_YELLOW}Ollama (Local Models){Colors.RESET}")
            if self.ollama_manager.check_ollama_available():
                installed_models = self.ollama_manager.get_installed_models()
                gemini_models = [m for m in installed_models if m.startswith(('gemma', 'gemini'))]
                if gemini_models:
                    print(f"{Colors.WHITE}  Models:   {Colors.BRIGHT_YELLOW}{', '.join(gemini_models[:3])}{Colors.RESET}")
                    if len(gemini_models) > 3:
                        print(f"{Colors.WHITE}            {Colors.BRIGHT_YELLOW}... and {len(gemini_models) - 3} more{Colors.RESET}")
                else:
                    print(f"{Colors.WHITE}  Models:   {Colors.BRIGHT_YELLOW}No Gemini models installed{Colors.RESET}")
            else:
                print(f"{Colors.WHITE}  Status:   {Colors.BRIGHT_YELLOW}Ollama not available{Colors.RESET}")
        else:
            print(f"{Colors.WHITE}  Provider: {Colors.BRIGHT_YELLOW}Google Official API{Colors.RESET}")
            if model:
                model_info = get_gemini_model_info(model)
                model_name = model_info.get('name', model)
                print(f"{Colors.WHITE}  Model:    {Colors.BRIGHT_YELLOW}{model_name}{Colors.RESET}")
        
        print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'─' * 50}{Colors.RESET}\n")
