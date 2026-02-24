#!/usr/bin/env python3
"""
Ultimate Zero-Configuration AI Agent Runner
Usage: python3 run.py "your instruction here"

This script automatically:
1. Detects if running in virtual environment
2. Creates virtual environment if needed
3. Installs all dependencies automatically
4. Restarts itself in the virtual environment
5. Prompts for model selection (Ollama or Google API)
6. Runs the AI agent with the provided instruction
"""

import sys
import os
import subprocess
import platform
import shutil
from pathlib import Path

# Global constants
VENV_DIR = "venv"
VENV_RESTART_FLAG = "--__venv_restarted__"

def is_in_virtual_environment():
    """Check if currently running in a virtual environment"""
    return (
        hasattr(sys, 'real_prefix') or 
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        os.getenv('VIRTUAL_ENV') is not None
    )

def get_venv_python_path():
    """Get the Python executable path in the virtual environment"""
    project_root = Path(__file__).parent
    venv_path = project_root / VENV_DIR
    
    if not venv_path.exists():
        return None
    
    if platform.system() == "Windows":
        python_exe = venv_path / "Scripts" / "python.exe"
        if not python_exe.exists():
            python_exe = venv_path / "Scripts" / "pythonw.exe"
    else:
        python_exe = venv_path / "bin" / "python"
        if not python_exe.exists():
            python_exe = venv_path / "bin" / "python3"
    
    return str(python_exe) if python_exe.exists() else None

def check_venv_prerequisites():
    """Check if virtual environment creation prerequisites are met"""
    print("Checking virtual environment prerequisites...")
    
    # Test if venv module is available
    try:
        import venv
        print("✓ venv module is available")
        return True
    except ImportError:
        print("✗ venv module is not available")
        return False

def create_virtual_environment():
    """Create a virtual environment with robust error handling"""
    project_root = Path(__file__).parent
    venv_path = project_root / VENV_DIR
    
    print(f"Creating virtual environment at {venv_path}...")
    
    # Remove existing venv if it exists and appears broken
    if venv_path.exists():
        venv_python = get_venv_python_path()
        if venv_python:
            try:
                # Test if existing venv works
                result = subprocess.run([venv_python, "--version"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    print("Existing virtual environment appears broken, recreating...")
                    shutil.rmtree(venv_path)
                else:
                    print("Virtual environment already exists and is functional")
                    return True
            except Exception:
                print("Existing virtual environment appears broken, recreating...")
                shutil.rmtree(venv_path)
        else:
            print("Removing incomplete virtual environment...")
            shutil.rmtree(venv_path)
    
    try:
        # Create virtual environment
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            
            # Handle specific error cases
            if "ensurepip is not available" in error_msg or "python3-venv" in error_msg:
                print("✗ Virtual environment creation failed: python3-venv package not installed")
                print()
                print("To fix this issue, run one of the following commands:")
                print(f"  sudo apt install python3.{sys.version_info.minor}-venv")
                print("  # or for Ubuntu/Debian systems:")
                print("  sudo apt install python3-venv")
                print()
                print("After installing the package, run this script again.")
                return False
            elif "Permission denied" in error_msg:
                print("✗ Permission denied when creating virtual environment")
                print("Check that you have write permissions to the project directory")
                return False
            else:
                print(f"✗ Failed to create virtual environment: {error_msg}")
                print("Full error details:")
                print(f"  Return code: {result.returncode}")
                print(f"  Stderr: {result.stderr}")
                print(f"  Stdout: {result.stdout}")
                return False
        
        print("✓ Virtual environment created successfully")
        return True
        
    except subprocess.TimeoutExpired:
        print("✗ Virtual environment creation timed out")
        return False
    except Exception as e:
        print(f"✗ Error creating virtual environment: {e}")
        return False

def restart_in_venv():
    """Restart the current script in the virtual environment with robust error handling"""
    venv_python = get_venv_python_path()
    if not venv_python:
        print("Error: Could not find virtual environment Python executable")
        return False
    
    # Add restart flag to prevent infinite loops
    new_argv = [venv_python, str(__file__), VENV_RESTART_FLAG] + sys.argv[1:]
    
    print(f"Restarting in virtual environment: {venv_python}")
    
    try:
        # Use os.execv to replace current process
        # This is more reliable than subprocess on all platforms
        os.execv(venv_python, new_argv)
    except OSError as e:
        print(f"OS error restarting in virtual environment: {e}")
        print("This might be due to permissions or antivirus software.")
        return False
    except Exception as e:
        print(f"Unexpected error restarting in virtual environment: {e}")
        return False
    
    # This should never be reached if execv succeeds
    return True

def install_dependencies():
    """Install all dependencies in the virtual environment with enhanced error handling"""
    project_root = Path(__file__).parent
    venv_python = get_venv_python_path()
    
    if not venv_python:
        print("Error: Virtual environment Python not found")
        return False
    
    print("Installing dependencies...")
    
    # Check network connectivity first
    try:
        import socket
        socket.create_connection(("pypi.org", 443), timeout=10)
        print("✓ Network connectivity OK")
    except Exception as e:
        print(f"Warning: Network connectivity issue: {e}")
        print("Dependency installation may fail without internet access.")
    
    # Upgrade pip first with retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"Retry {attempt + 1}/{max_retries} upgrading pip...")
            else:
                print("Upgrading pip...")
            
            result = subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"],
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✓ pip upgraded")
                break
            else:
                if attempt == max_retries - 1:
                    print(f"pip upgrade failed after {max_retries} attempts: {result.stderr}")
                    print("Continuing with current pip version...")
                else:
                    print(f"pip upgrade attempt {attempt + 1} failed, retrying...")
        except subprocess.TimeoutExpired:
            if attempt == max_retries - 1:
                print("pip upgrade timed out, continuing with current pip version...")
            else:
                print("pip upgrade timed out, retrying...")
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"pip upgrade error: {e}")
                print("Continuing with current pip version...")
            else:
                print(f"pip upgrade error: {e}, retrying...")
    
    # Install from requirements files if they exist
    requirements_files = [
        project_root / "requirements-core.txt",
        project_root / "requirements.txt"  # fallback to original
    ]
    
    for requirements_file in requirements_files:
        if requirements_file.exists():
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        print(f"Retry {attempt + 1}/{max_retries} installing {requirements_file.name}...")
                    else:
                        print(f"Installing from {requirements_file.name}...")
                    
                    result = subprocess.run([venv_python, "-m", "pip", "install", "-r", str(requirements_file)],
                                          capture_output=True, text=True, timeout=600)
                    if result.returncode == 0:
                        print(f"✓ {requirements_file.name} installed")
                        
                        # If we successfully installed core requirements, we're done
                        if requirements_file.name == "requirements-core.txt":
                            print("✓ Core dependencies installed successfully")
                            print("Note: Optional ML/AI dependencies can be installed later with:")
                            print("  pip install -r requirements-optional.txt")
                            return True  # Success, exit the function
                        break
                    else:
                        error_msg = result.stderr.strip()
                        if attempt == max_retries - 1:
                            print(f"{requirements_file.name} installation failed after {max_retries} attempts: {error_msg}")
                            
                            # Provide helpful error messages
                            if "Permission denied" in error_msg:
                                print("Permission denied. Check antivirus software or file permissions.")
                            elif "Could not find a version" in error_msg:
                                print("Package version conflict. Check requirements file compatibility.")
                            elif "Network is unreachable" in error_msg or "Connection failed" in error_msg:
                                print("Network error. Check internet connection.")
                            else:
                                print("See error message above for details.")
                            
                            # If this was requirements-core.txt that failed, return False
                            # If it was requirements.txt that failed, we can continue (it's optional)
                            if requirements_file.name == "requirements-core.txt":
                                return False
                            else:
                                print("Continuing without optional dependencies...")
                                return True  # Continue without optional deps
                        else:
                            print(f"{requirements_file.name} attempt {attempt + 1} failed, retrying...")
                except subprocess.TimeoutExpired:
                    if attempt == max_retries - 1:
                        print(f"{requirements_file.name} installation timed out")
                        if requirements_file.name == "requirements-core.txt":
                            return False
                        else:
                            print("Continuing without optional dependencies...")
                            return True  # Continue without optional deps
                    else:
                        print(f"{requirements_file.name} installation timed out, retrying...")
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"{requirements_file.name} installation error: {e}")
                        if requirements_file.name == "requirements-core.txt":
                            return False
                        else:
                            print("Continuing without optional dependencies...")
                            return True  # Continue without optional deps
                    else:
                        print(f"{requirements_file.name} installation error: {e}, retrying...")
    
    # Install project in editable mode if pyproject.toml exists
    pyproject_file = project_root / "pyproject.toml"
    if pyproject_file.exists():
        try:
            print("Installing project in editable mode...")
            result = subprocess.run([venv_python, "-m", "pip", "install", "-e", str(project_root)],
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✓ project installed")
            else:
                print(f"project installation warning: {result.stderr}")
                print("Project installation failed, but dependencies may still work")
        except subprocess.TimeoutExpired:
            print("project installation timed out")
            print("Project installation failed, but dependencies may still work")
        except Exception as e:
            print(f"project installation error: {e}")
            print("Project installation failed, but dependencies may still work")
    
    return True

def bootstrap_environment():
    """Bootstrap the environment - create venv and install dependencies"""
    print("Bootstrapping environment...")
    
    # Check prerequisites first
    if not check_venv_prerequisites():
        print()
        print("Virtual environment prerequisites not met.")
        print("This is likely because the python3-venv package is not installed.")
        print()
        print("To fix this issue, run one of the following commands:")
        print(f"  sudo apt install python3.{sys.version_info.minor}-venv")
        print("  # or for Ubuntu/Debian systems:")
        print("  sudo apt install python3-venv")
        print()
        print("After installing the package, run this script again.")
        return False
    
    # Create virtual environment
    if not create_virtual_environment():
        print("Failed to create virtual environment")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies")
        return False
    
    print("✓ Environment bootstrap complete")
    return True

def show_help():
    """Show help message"""
    print("VEXIS-1.1 AI Agent Runner")
    print("=" * 50)
    print("Usage: python3 run.py \"your instruction here\"")
    print()
    print("This script automatically handles:")
    print("  • Virtual environment creation and management")
    print("  • Dependency installation")
    print("  • Model selection (Ollama or Google API)")
    print("  • Cross-platform compatibility")
    print("  • Self-bootstrapping")
    print()
    print("Model Options:")
    print("  • Ollama: Local models via Ollama (requires ollama signin)")
    print("  • Google API: Official Google Gemini API (requires API key)")
    print("    - Gemini 3 Flash: Fast and cost-effective")
    print("    - Gemini 3.1 Pro: Advanced reasoning for complex tasks")
    print()
    print("Examples:")
    print("  python3 run.py \"Take a screenshot\"")
    print("  python3 run.py \"Open a web browser and search for AI\"")
    print()
    print("Options:")
    print("  --help, -h          Show this help message")
    print("  --debug             Enable debug mode")
    print()
    print("Virtual Environment:")
    print("  Automatically creates and uses './venv' directory")
    print("  All dependencies are isolated within the virtual environment")
    print("  No manual setup required - just run and go!")

def check_ollama_login():
    """Check if Ollama is logged in and prompt for sign-in if needed"""
    from src.ai_agent.utils.interactive_menu import Colors, success_message, error_message, warning_message
    
    try:
        # Check if Ollama is available
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            error_message("Ollama is not installed or not in PATH")
            print(f"{Colors.BRIGHT_CYAN}Please install Ollama first: https://ollama.com/{Colors.RESET}")
            return False
        
        # Check if signed in
        result = subprocess.run(["ollama", "whoami"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0 or "not signed in" in result.stderr.lower():
            warning_message("Ollama is available but you are not signed in.")
            print(f"{Colors.CYAN}Running 'ollama signin' to sign in...{Colors.RESET}")
            
            # Prompt for sign-in
            signin_result = subprocess.run(["ollama", "signin"], 
                                          timeout=120)
            if signin_result.returncode == 0:
                success_message("Ollama sign-in completed")
                return True
            else:
                error_message("Ollama sign-in failed")
                return False
        else:
            success_message("Ollama is signed in")
            return True
            
    except subprocess.TimeoutExpired:
        error_message("Ollama command timed out")
        return False
    except FileNotFoundError:
        error_message("Ollama is not installed")
        print(f"{Colors.BRIGHT_CYAN}Please install Ollama first: https://ollama.com/{Colors.RESET}")
        return False
    except Exception as e:
        error_message(f"Error checking Ollama: {e}")
        return False

def prompt_for_google_api_key():
    """Prompt user for Google API key and handle saving"""
    import getpass
    from src.ai_agent.utils.interactive_menu import Colors, success_message, error_message, warning_message
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}Google API Key Setup{Colors.RESET}")
    print(f"{Colors.CYAN}{'-' * 25}{Colors.RESET}")
    print(f"{Colors.WHITE}To use Google's official Gemini API, you need an API key.{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}You can get one from: https://aistudio.google.com/app/apikey{Colors.RESET}")
    print()
    
    while True:
        try:
            api_key = getpass.getpass(f"{Colors.YELLOW}Enter your Google API key (or press Enter to cancel):{Colors.RESET} ")
            if not api_key.strip():
                warning_message("No API key provided. Skipping Google API setup.")
                return None
            
            # Basic validation (Google API keys are typically 39 characters starting with 'AIza')
            if len(api_key) < 20:
                error_message("API key seems too short. Please check your key.")
                continue
            
            # Ask if user wants to save the key
            save_key = input(f"{Colors.CYAN}Save this API key for future use? (y/n):{Colors.RESET} ").lower().strip()
            should_save = save_key.startswith('y')
            
            return api_key, should_save
            
        except KeyboardInterrupt:
            print(f"\n{Colors.BRIGHT_YELLOW}Operation cancelled.{Colors.RESET}")
            return None
        except Exception as e:
            error_message(f"Error reading input: {e}")
            return None

def select_google_model():
    """Prompt user to select Google model with interactive menu"""
    from src.ai_agent.utils.settings_manager import get_settings_manager
    from src.ai_agent.utils.interactive_menu import InteractiveMenu
    
    settings_manager = get_settings_manager()
    current_model = settings_manager.get_google_model()
    
    menu = InteractiveMenu(
        "Select Gemini Model",
        "Choose your preferred Gemini model:"
    )
    
    menu.add_item(
        "Gemini 3 Flash",
        "Fast and efficient • Cost-effective for most tasks",
        "gemini-3-flash-preview",
        "🚀"
    )
    
    menu.add_item(
        "Gemini 3.1 Pro",
        "Advanced reasoning • Best for complex problem-solving",
        "gemini-3.1-pro-preview",
        "🧠"
    )
    
    menu.set_current_selection(current_model)
    selected_model = menu.show()
    
    if selected_model is None:
        return current_model
    
    settings_manager.set_google_model(selected_model)
    return selected_model

def show_config_summary(provider: str, model: str = None):
    """Display a clean configuration summary"""
    from src.ai_agent.utils.interactive_menu import Colors
    
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{'─' * 50}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}✓ Configuration Complete{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'─' * 50}{Colors.RESET}")
    
    if provider == "ollama":
        print(f"{Colors.WHITE}  Provider: {Colors.BRIGHT_YELLOW}Ollama (Local Models){Colors.RESET}")
    else:
        print(f"{Colors.WHITE}  Provider: {Colors.BRIGHT_YELLOW}Google Official API{Colors.RESET}")
        if model:
            model_name = "Gemini 3 Flash" if model == "gemini-3-flash-preview" else "Gemini 3.1 Pro"
            print(f"{Colors.WHITE}  Model:    {Colors.BRIGHT_YELLOW}{model_name}{Colors.RESET}")
    
    
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'─' * 50}{Colors.RESET}\n")

def configure_google_provider():
    """Configure Google provider with API key and model selection"""
    from src.ai_agent.utils.settings_manager import get_settings_manager
    from src.ai_agent.utils.interactive_menu import Colors
    
    settings_manager = get_settings_manager()
    
    # Check if API key already exists
    if not settings_manager.has_google_api_key():
        # Prompt for API key
        result = prompt_for_google_api_key()
        if result is None:
            return None, None
        
        api_key, should_save = result
        settings_manager.set_google_api_key(api_key, should_save)
    
    
    # Select model
    model = select_google_model()
    if model is None:
        model = settings_manager.get_google_model()
    
    settings_manager.set_preferred_provider("google")
    return "google", model

def configure_ollama_provider():
    """Configure Ollama provider"""
    from src.ai_agent.utils.settings_manager import get_settings_manager
    
    settings_manager = get_settings_manager()
    
    if not check_ollama_login():
        return None
    
    settings_manager.set_preferred_provider("ollama")
    return "ollama"

def select_model_provider():
    """Main configuration screen for model provider selection"""
    from src.ai_agent.utils.settings_manager import get_settings_manager
    from src.ai_agent.utils.interactive_menu import (
        InteractiveMenu, Colors
    )
    
    settings_manager = get_settings_manager()
    current_provider = settings_manager.get_preferred_provider()
    
    # Clear screen for clean configuration display
    print(f"\033[2J\033[H", end="")
    
    # Show header
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'═' * 50}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}     VEXIS-1.1 AI Agent Configuration{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'═' * 50}{Colors.RESET}\n")
    
    # Provider selection
    menu = InteractiveMenu(
        "Select AI Provider",
        "Choose how you want to run AI models:"
    )
    
    menu.add_item(
        "Ollama (Local)",
        "Run models locally via Ollama • Privacy-focused",
        "ollama",
        "🦊"
    )
    
    menu.add_item(
        "Google Official API",
        "Use Google's cloud Gemini models • Requires API key",
        "google",
        "🌐"
    )
    
    menu.set_current_selection(current_provider)
    selected_provider = menu.show()
    
    if selected_provider is None:
        # User cancelled - use current settings
        if current_provider == "google":
            model = settings_manager.get_google_model()
            show_config_summary(current_provider, model)
        else:
            show_config_summary(current_provider)
        return current_provider
    
    
    # Handle provider selection
    if selected_provider == "ollama":
        result = configure_ollama_provider()
        if result is None:
            # Failed - retry configuration
            return select_model_provider()
        show_config_summary("ollama")
        return "ollama"
        
    elif selected_provider == "google":
        provider, model = configure_google_provider()
        if provider is None:
            # User cancelled API key entry - retry
            return select_model_provider()
        show_config_summary(provider, model)
        return "google"

def main():
    """Main entry point"""
    # Check for help flag first
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit(0)
    
    # Check if we've already restarted in venv
    if VENV_RESTART_FLAG in sys.argv:
        # Remove the restart flag for clean processing
        sys.argv.remove(VENV_RESTART_FLAG)
        print("✓ Running in virtual environment")
    else:
        # Not in venv or not restarted yet
        if not is_in_virtual_environment():
            print("Not in virtual environment")
            
            # Check if venv exists and is functional
            venv_python = get_venv_python_path()
            if venv_python:
                try:
                    result = subprocess.run([venv_python, "--version"], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        print("Virtual environment found, restarting...")
                        restart_in_venv()
                        return  # This should never execute if restart works
                except Exception:
                    pass
            
            # No working venv found, create one
            if bootstrap_environment():
                print("Restarting in new virtual environment...")
                restart_in_venv()
                return  # This should never execute if restart works
            else:
                print("Failed to bootstrap environment")
                sys.exit(1)
        else:
            print("✓ Already in virtual environment")
    
    # At this point, we're running in a virtual environment
    # Add src to Python path
    current_dir = Path(__file__).parent
    src_dir = current_dir / "src"
    sys.path.insert(0, str(src_dir))
    
    # Validate arguments
    if len(sys.argv) < 2:
        print("Usage: python3 run.py \"your instruction here\"")
        print("Example: python3 run.py \"Take a screenshot\"")
        print("Use --help for more options")
        sys.exit(1)
    
    # Filter out flags to get the actual instruction
    instruction_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    instruction = " ".join(instruction_args)
    
    if not instruction:
        print("No instruction provided")
        print("Usage: python3 run.py \"your instruction here\"")
        sys.exit(1)
    
    # Check for debug mode
    debug_mode = "--debug" in sys.argv
    
    # Model selection - only prompt if not using --no-prompt flag
    if "--no-prompt" not in sys.argv:
        selected_provider = select_model_provider()
        print(f"\nUsing provider: {selected_provider}")
    else:
        from src.ai_agent.utils.settings_manager import get_settings_manager
        settings_manager = get_settings_manager()
        selected_provider = settings_manager.get_preferred_provider()
        print(f"\nUsing saved provider preference: {selected_provider}")
    
    print(f"\nAI Agent executing: {instruction}")
    
    try:
        from ai_agent.user_interface.two_phase_app import TwoPhaseAIAgent
        
        # Update config with selected provider
        config_path = current_dir / "config.yaml"
        agent = TwoPhaseAIAgent(config_path=str(config_path) if config_path.exists() else None)
        
        # Update the vision client configuration with the selected provider
        if hasattr(agent, 'engine') and hasattr(agent.engine, 'model_runner'):
            model_runner = agent.engine.model_runner
            if hasattr(model_runner, 'vision_client'):
                # Update the vision client config
                from src.ai_agent.utils.settings_manager import get_settings_manager
                settings_manager = get_settings_manager()
                
                # Reload config with updated provider settings
                updated_config = model_runner.config.copy()
                updated_config['preferred_provider'] = selected_provider
                updated_config['google_api_key'] = settings_manager.get_google_api_key()
                updated_config['google_model'] = settings_manager.get_google_model()
                
                # Reinitialize vision client with updated config
                model_runner.vision_client.config = updated_config
        
        # Run the instruction
        options = {"debug": debug_mode}
        result = agent.run(instruction, options)
        
        if result:
            print("\n✓ Task completed successfully")
        else:
            print("\n✗ Task failed")
            sys.exit(1)
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("This suggests a dependency issue. The virtual environment may not be set up correctly.")
        print("Try deleting the 'venv' directory and running again.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if debug_mode:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
