#!/usr/bin/env python3
"""
Ultimate Zero-Configuration AI Agent Runner
Usage: python3 run.py "your instruction here"

This script automatically:
1. Detects if running in virtual environment
2. Creates virtual environment if needed
3. Installs all dependencies automatically
4. Restarts itself in the virtual environment
5. Runs the AI agent with the provided instruction
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
            print(f"Failed to create virtual environment: {result.stderr}")
            return False
        
        print("✓ Virtual environment created successfully")
        return True
        
    except subprocess.TimeoutExpired:
        print("Virtual environment creation timed out")
        return False
    except Exception as e:
        print(f"Error creating virtual environment: {e}")
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
    
    # Install from requirements.txt if it exists
    requirements_file = project_root / "requirements.txt"
    if requirements_file.exists():
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"Retry {attempt + 1}/{max_retries} installing requirements.txt...")
                else:
                    print("Installing from requirements.txt...")
                
                result = subprocess.run([venv_python, "-m", "pip", "install", "-r", str(requirements_file)],
                                      capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    print("✓ requirements.txt installed")
                    break
                else:
                    error_msg = result.stderr.strip()
                    if attempt == max_retries - 1:
                        print(f"requirements.txt installation failed after {max_retries} attempts: {error_msg}")
                        
                        # Provide helpful error messages
                        if "Permission denied" in error_msg:
                            print("Permission denied. Check antivirus software or file permissions.")
                        elif "Could not find a version" in error_msg:
                            print("Package version conflict. Check requirements.txt compatibility.")
                        elif "Network is unreachable" in error_msg or "Connection failed" in error_msg:
                            print("Network error. Check internet connection.")
                        else:
                            print("See error message above for details.")
                        return False
                    else:
                        print(f"requirements.txt attempt {attempt + 1} failed, retrying...")
            except subprocess.TimeoutExpired:
                if attempt == max_retries - 1:
                    print("requirements.txt installation timed out")
                    return False
                else:
                    print("requirements.txt installation timed out, retrying...")
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"requirements.txt installation error: {e}")
                    return False
                else:
                    print(f"requirements.txt installation error: {e}, retrying...")
    
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
    print("VEXIS-1 AI Agent Runner")
    print("=" * 50)
    print("Usage: python3 run.py \"your instruction here\"")
    print()
    print("This script automatically handles:")
    print("  • Virtual environment creation and management")
    print("  • Dependency installation")
    print("  • Cross-platform compatibility")
    print("  • Self-bootstrapping")
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
    
    print(f"\nAI Agent executing: {instruction}")
    
    try:
        from ai_agent.user_interface.two_phase_app import TwoPhaseAIAgent
        
        # Initialize the agent
        config_path = current_dir / "config.yaml"
        agent = TwoPhaseAIAgent(config_path=str(config_path) if config_path.exists() else None)
        
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
