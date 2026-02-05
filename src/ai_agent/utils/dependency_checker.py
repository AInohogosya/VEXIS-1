#!/usr/bin/env python3
"""
Dependency Checker and Auto-Installer
Ensures all required dependencies are available before running the AI Agent
"""

import sys
import subprocess
import platform
import importlib
import os
import time
import socket
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class DependencyChecker:
    """Comprehensive dependency checking and auto-installation system"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.requirements_file = project_root / "requirements.txt"
        self.pyproject_file = project_root / "pyproject.toml"
        
        # Core dependencies that must be available
        self.core_dependencies = {
            "PIL": "Pillow>=10.0.0",
            "pyautogui": "pyautogui>=0.9.54", 
            "requests": "requests>=2.31.0",
            "cv2": "opencv-python>=4.8.0",
            "numpy": "numpy>=1.24.0",
            "pynput": "pynput>=1.7.6",
            "openai": "openai>=1.0.0",
            "anthropic": "anthropic>=0.7.0",
            "transformers": "transformers>=4.35.0",
            "torch": "torch>=2.1.0",
            "cryptography": "cryptography>=41.0.0",
            "pydantic": "pydantic>=2.0.0",
            "structlog": "structlog>=23.0.0",
            "rich": "rich>=13.0.0",
            "yaml": "PyYAML>=6.0.0",  # yaml module imports as PyYAML package
            "ollama": "ollama>=0.1.0",  # Add missing ollama dependency
        }
        
        # Platform-specific dependencies
        self.platform_dependencies = {
            "darwin": {
                "objc": "pyobjc-framework-Cocoa>=9.0"
            },
            "win32": {
                "win32api": "pywin32>=306"
            },
            "linux": {
                "Xlib": "python-xlib>=0.33"
            }
        }
        
        # System packages that might need installation
        self.system_packages = {
            "darwin": [
                "xcode-select",  # For Xcode command line tools
            ],
            "win32": [
                # Windows usually has these pre-installed
            ],
            "linux": {
                "debian": [
                    "python3-dev", "python3-pip", "python3-venv",
                    "scrot", "python3-tk", "xvfb", "x11-utils"
                ],
                "redhat": [
                    "python3-devel", "python3-pip", "scrot", 
                    "tkinter", "xorg-x11-server-Xvfb"
                ]
            }
        }

    def check_python_version(self) -> Tuple[bool, str]:
        """Check if Python version meets requirements"""
        required_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version >= required_version:
            return True, f"Python {current_version[0]}.{current_version[1]} ✓"
        else:
            return False, f"Python {current_version[0]}.{current_version[1]} (required >=3.8) ✗"
    
    def check_pip_version(self) -> Tuple[bool, str]:
        """Check if pip is available and reasonably up-to-date"""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                version_str = result.stdout.strip()
                return True, f"pip {version_str.split()[1]} ✓"
            else:
                return False, "pip not working ✗"
        except Exception as e:
            return False, f"pip check failed: {str(e)} ✗"
    
    def upgrade_pip(self) -> Tuple[bool, str]:
        """Upgrade pip to latest version"""
        try:
            print("🔄 Upgrading pip to latest version...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                return True, "pip upgraded successfully"
            else:
                return False, f"pip upgrade failed: {result.stderr}"
        except Exception as e:
            return False, f"pip upgrade error: {str(e)}"
    
    def check_network_connectivity(self) -> Tuple[bool, str]:
        """Check if network connectivity is available"""
        try:
            # Try to connect to PyPI
            socket.create_connection(("pypi.org", 443), timeout=10)
            return True, "Network connectivity OK ✓"
        except socket.gaierror:
            return False, "DNS resolution failed - check internet connection ✗"
        except socket.timeout:
            return False, "Network timeout - check internet connection ✗"
        except Exception as e:
            return False, f"Network check failed: {str(e)} ✗"
    
    def check_virtual_env(self) -> Tuple[bool, str]:
        """Check if running in virtual environment"""
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            venv_path = sys.prefix
            return True, f"Virtual environment: {venv_path} ✓"
        else:
            return False, "Not in virtual environment (system Python) ⚠️"
    
    def get_venv_python_executable(self) -> Optional[str]:
        """Get the Python executable path for the virtual environment"""
        # Check if we're currently in a virtual environment
        venv_ok, venv_msg = self.check_virtual_env()
        if venv_ok:
            return sys.executable
        
        # Check if there's a venv directory in the project root
        venv_path = self.project_root / "venv"
        if venv_path.exists() and venv_path.is_dir():
            # Try different possible Python executable paths
            if sys.platform == "win32":
                python_exe = venv_path / "Scripts" / "python.exe"
                pythonw_exe = venv_path / "Scripts" / "pythonw.exe"
                if python_exe.exists():
                    return str(python_exe)
                elif pythonw_exe.exists():
                    return str(pythonw_exe)
            else:
                python_exe = venv_path / "bin" / "python"
                if python_exe.exists():
                    return str(python_exe)
        
        return None
    
    def get_venv_pip_executable(self) -> Optional[str]:
        """Get the pip executable path for the virtual environment"""
        venv_python = self.get_venv_python_executable()
        if venv_python:
            # Use python -m pip instead of direct pip path for better reliability
            return [venv_python, "-m", "pip"]
        
        return None
    
    def create_virtual_environment(self, force: bool = False) -> Tuple[bool, str]:
        """Create a virtual environment if not in one"""
        if not force:
            venv_ok, venv_msg = self.check_virtual_env()
            if venv_ok:
                return True, "Already in virtual environment"
        
        venv_path = self.project_root / "venv"
        
        if venv_path.exists() and not force:
            return True, f"Virtual environment already exists at {venv_path}"
        
        try:
            print(f"🔧 Creating virtual environment at {venv_path}...")
            result = subprocess.run(
                [sys.executable, "-m", "venv", str(venv_path)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                print(f"✅ Virtual environment created successfully")
                print(f"💡 Activate it with: source {venv_path}/bin/activate")
                return True, f"Virtual environment created at {venv_path}"
            else:
                return False, f"Failed to create virtual environment: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "Virtual environment creation timed out"
        except Exception as e:
            return False, f"Error creating virtual environment: {str(e)}"

    def check_import(self, module_name: str) -> bool:
        """Check if a module can be imported"""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
        except PermissionError:
            # Handle permission errors (e.g., when rich tries to access cwd)
            return False
        except OSError:
            # Handle other OS-level errors during import
            return False
        except Exception:
            # Handle any other unexpected import errors
            return False

    def get_package_version(self, module_name: str) -> Optional[str]:
        """Get version of an installed package"""
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, '__version__'):
                return module.__version__
            elif hasattr(module, 'version'):
                return module.version
            else:
                return "unknown"
        except ImportError:
            return None

    def check_core_dependencies(self) -> Dict[str, Tuple[bool, str]]:
        """Check all core dependencies"""
        results = {}
        
        print("🔍 Checking core Python dependencies...")
        
        for module, package in self.core_dependencies.items():
            if self.check_import(module):
                version = self.get_package_version(module)
                results[module] = (True, f"{package} ({version}) ✓")
            else:
                results[module] = (False, f"{package} ✗")
        
        return results

    def check_platform_dependencies(self) -> Dict[str, Tuple[bool, str]]:
        """Check platform-specific dependencies"""
        results = {}
        current_platform = sys.platform
        
        print(f"🔍 Checking {current_platform} platform dependencies...")
        
        if current_platform in self.platform_dependencies:
            for module, package in self.platform_dependencies[current_platform].items():
                if self.check_import(module):
                    version = self.get_package_version(module)
                    results[module] = (True, f"{package} ({version}) ✓")
                else:
                    results[module] = (False, f"{package} ✗")
        
        return results

    def install_package(self, package: str, retries: int = 3, use_venv: bool = True) -> Tuple[bool, str]:
        """Install a package using pip with retry mechanism"""
        # Determine which Python/pip to use
        pip_cmd = None
        python_exe = None
        
        if use_venv:
            venv_pip = self.get_venv_pip_executable()
            if venv_pip:
                pip_cmd = venv_pip
                python_exe = self.get_venv_python_executable()
                print(f"🔧 Using virtual environment: {python_exe}")
            else:
                print("🔧 No virtual environment found, using system Python")
        
        if not pip_cmd:
            # Fall back to system Python
            pip_cmd = [sys.executable, "-m", "pip"]
        
        for attempt in range(retries):
            try:
                if attempt > 0:
                    print(f"🔄 Retry {attempt + 1}/{retries} for {package}...")
                else:
                    print(f"📦 Installing {package}...")
                
                result = subprocess.run(
                    pip_cmd + ["install", package],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    return True, f"Successfully installed {package}"
                else:
                    error_msg = result.stderr.strip()
                    # Check for common issues and provide specific guidance
                    if "Permission denied" in error_msg:
                        if use_venv and python_exe:
                            return False, f"Permission denied installing {package}. Virtual environment may have permission issues."
                        else:
                            return False, f"Permission denied installing {package}. Try using --user flag or virtual environment."
                    elif "Could not find a version" in error_msg:
                        return False, f"Package {package} not found or version incompatible."
                    elif "Network is unreachable" in error_msg or "Connection failed" in error_msg:
                        return False, f"Network error installing {package}. Check internet connection."
                    elif attempt == retries - 1:
                        return False, f"Failed to install {package} after {retries} attempts: {error_msg}"
                    else:
                        time.sleep(2)  # Wait before retry
                        continue
                        
            except subprocess.TimeoutExpired:
                if attempt == retries - 1:
                    return False, f"Installation of {package} timed out after {retries} attempts"
                time.sleep(5)  # Wait longer before retry
                continue
            except Exception as e:
                if attempt == retries - 1:
                    return False, f"Error installing {package}: {str(e)}"
                time.sleep(2)
                continue
        
        return False, f"Failed to install {package} after {retries} attempts"

    def install_requirements_file(self, retries: int = 2, use_venv: bool = True) -> Tuple[bool, str]:
        """Install all dependencies from requirements.txt with retry"""
        if not self.requirements_file.exists():
            return False, "requirements.txt not found"
        
        # Determine which Python/pip to use
        pip_cmd = None
        python_exe = None
        
        if use_venv:
            venv_pip = self.get_venv_pip_executable()
            if venv_pip:
                pip_cmd = venv_pip
                python_exe = self.get_venv_python_executable()
                print(f"🔧 Using virtual environment: {python_exe}")
            else:
                print("🔧 No virtual environment found, using system Python")
        
        if not pip_cmd:
            # Fall back to system Python
            pip_cmd = [sys.executable, "-m", "pip"]
        
        for attempt in range(retries):
            try:
                if attempt > 0:
                    print(f"🔄 Retry {attempt + 1}/{retries} for requirements.txt...")
                else:
                    print("📦 Installing dependencies from requirements.txt...")
                
                result = subprocess.run(
                    pip_cmd + ["install", "-r", str(self.requirements_file)],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                
                if result.returncode == 0:
                    return True, "Successfully installed requirements.txt"
                else:
                    error_msg = result.stderr.strip()
                    if attempt == retries - 1:
                        return False, f"Failed to install requirements.txt after {retries} attempts: {error_msg}"
                    else:
                        time.sleep(3)  # Wait before retry
                        continue
                        
            except subprocess.TimeoutExpired:
                if attempt == retries - 1:
                    return False, f"Installation of requirements.txt timed out after {retries} attempts"
                time.sleep(5)
                continue
            except Exception as e:
                if attempt == retries - 1:
                    return False, f"Error installing requirements.txt: {str(e)}"
                time.sleep(3)
                continue
        
        return False, f"Failed to install requirements.txt after {retries} attempts"

    def install_project(self, retries: int = 2, use_venv: bool = True) -> Tuple[bool, str]:
        """Install the project in editable mode with retry"""
        if not self.pyproject_file.exists():
            return False, "pyproject.toml not found"
        
        # Determine which Python/pip to use
        pip_cmd = None
        python_exe = None
        
        if use_venv:
            venv_pip = self.get_venv_pip_executable()
            if venv_pip:
                pip_cmd = venv_pip
                python_exe = self.get_venv_python_executable()
                print(f"🔧 Using virtual environment: {python_exe}")
            else:
                print("🔧 No virtual environment found, using system Python")
        
        if not pip_cmd:
            # Fall back to system Python
            pip_cmd = [sys.executable, "-m", "pip"]
        
        for attempt in range(retries):
            try:
                if attempt > 0:
                    print(f"🔄 Retry {attempt + 1}/{retries} for project installation...")
                else:
                    print("📦 Installing project in editable mode...")
                
                result = subprocess.run(
                    pip_cmd + ["install", "-e", str(self.project_root)],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    return True, "Successfully installed project"
                else:
                    error_msg = result.stderr.strip()
                    if attempt == retries - 1:
                        return False, f"Failed to install project after {retries} attempts: {error_msg}"
                    else:
                        time.sleep(3)
                        continue
                        
            except subprocess.TimeoutExpired:
                if attempt == retries - 1:
                    return False, f"Project installation timed out after {retries} attempts"
                time.sleep(5)
                continue
            except Exception as e:
                if attempt == retries - 1:
                    return False, f"Error installing project: {str(e)}"
                time.sleep(3)
                continue
        
        return False, f"Failed to install project after {retries} attempts"

    def check_system_dependencies(self) -> Dict[str, Tuple[bool, str]]:
        """Check system-level dependencies"""
        results = {}
        current_platform = sys.platform
        
        print(f"🔍 Checking system dependencies for {current_platform}...")
        
        if current_platform == "linux":
            # Try to detect distribution
            try:
                with open("/etc/os-release", "r") as f:
                    os_release = f.read()
                    if "debian" in os_release.lower() or "ubuntu" in os_release.lower():
                        distro = "debian"
                    elif "rhel" in os_release.lower() or "centos" in os_release.lower() or "fedora" in os_release.lower():
                        distro = "redhat"
                    else:
                        distro = "unknown"
            except:
                distro = "unknown"
            
            if distro in self.system_packages["linux"]:
                for package in self.system_packages["linux"][distro]:
                    # Check if package is available
                    try:
                        result = subprocess.run(["which", package], capture_output=True)
                        if result.returncode == 0:
                            results[package] = (True, f"{package} ✓")
                        else:
                            results[package] = (False, f"{package} ✗")
                    except:
                        results[package] = (False, f"{package} ✗")
        
        elif current_platform == "darwin":
            for package in self.system_packages["darwin"]:
                try:
                    result = subprocess.run(["which", package], capture_output=True)
                    if result.returncode == 0:
                        results[package] = (True, f"{package} ✓")
                    else:
                        results[package] = (False, f"{package} ✗")
                except:
                    results[package] = (False, f"{package} ✗")
        
        return results

    def auto_install_missing(self, missing_deps: List[str]) -> bool:
        """Attempt to auto-install missing dependencies with enhanced error handling"""
        if not missing_deps:
            return True
        
        print(f"\n🔧 Found {len(missing_deps)} missing dependencies. Attempting auto-install...")
        
        # Check network connectivity first
        net_ok, net_msg = self.check_network_connectivity()
        if not net_ok:
            print(f"❌ {net_msg}")
            print("💡 Please check your internet connection and try again.")
            return False
        print(f"✅ {net_msg}")
        
        # Check and upgrade pip if needed
        pip_ok, pip_msg = self.check_pip_version()
        print(f"📦 {pip_msg}")
        
        if not pip_ok:
            print("🔄 Attempting to fix pip installation...")
            upgrade_ok, upgrade_msg = self.upgrade_pip()
            if upgrade_ok:
                print(f"✅ {upgrade_msg}")
            else:
                print(f"⚠️  {upgrade_msg}")
                print("💡 Continuing with current pip version...")
        
        # Check virtual environment
        venv_ok, venv_msg = self.check_virtual_env()
        print(f"🐍 {venv_msg}")
        
        use_venv = False
        if not venv_ok:
            print("⚠️  Running in system Python may cause permission issues.")
            print("💡 Consider using a virtual environment for better reliability.")
            
            # Offer to create virtual environment if there are permission issues
            if len(missing_deps) > 5:  # Many missing deps suggests system Python issues
                print("🤔 Would you like to create a virtual environment? (Recommended)")
                print("   This will avoid permission issues and isolate dependencies.")
                print("   The environment will be created at: ./venv")
                
                # Auto-create venv for better user experience
                try:
                    venv_success, venv_message = self.create_virtual_environment()
                    if venv_success:
                        print(f"✅ {venv_message}")
                        print("� Using created virtual environment for installation...")
                        use_venv = True
                    else:
                        print(f"❌ {venv_message}")
                        print("🔄 Continuing with system Python...")
                except Exception as e:
                    print(f"❌ Could not create virtual environment: {e}")
                    print("🔄 Continuing with system Python...")
            else:
                print("🔄 Installing with system Python (few dependencies missing)...")
        else:
            use_venv = True
            print("🔧 Using existing virtual environment for installation...")
        
        # First try to install from requirements.txt if it exists
        if self.requirements_file.exists():
            success, message = self.install_requirements_file(use_venv=use_venv)
            if success:
                print(f"✅ {message}")
                return True
            else:
                print(f"⚠️  {message}")
                print("🔄 Falling back to individual package installation...")
        
        # Fall back to individual package installation
        failed_packages = []
        for dep in missing_deps:
            if dep in self.core_dependencies:
                package = self.core_dependencies[dep]
            elif sys.platform in self.platform_dependencies:
                platform_deps = self.platform_dependencies[sys.platform]
                if dep in platform_deps:
                    package = platform_deps[dep]
                else:
                    package = dep
            else:
                package = dep
            
            success, message = self.install_package(package, use_venv=use_venv)
            if success:
                print(f"✅ {message}")
            else:
                print(f"❌ {message}")
                failed_packages.append(dep)
        
        # Install project in editable mode
        success, message = self.install_project(use_venv=use_venv)
        if success:
            print(f"✅ {message}")
        else:
            print(f"⚠️  {message}")
        
        if failed_packages:
            print(f"\n❌ Failed to install: {', '.join(failed_packages)}")
            print("💡 Try installing these manually:")
            for pkg in failed_packages:
                if pkg in self.core_dependencies:
                    print(f"   pip install {self.core_dependencies[pkg]}")
                else:
                    print(f"   pip install {pkg}")
            return False
        
        return True

    def run_full_check(self, auto_install: bool = True) -> bool:
        """Run comprehensive dependency check"""
        print("🚀 Starting dependency check for VEXIS-1 AI Agent\n")
        
        # Check Python version
        py_ok, py_msg = self.check_python_version()
        print(f"🐍 {py_msg}")
        if not py_ok:
            print("❌ Python version too old. Please upgrade to Python 3.8 or higher.")
            return False
        
        # Check core dependencies
        core_results = self.check_core_dependencies()
        missing_core = [mod for mod, (ok, _) in core_results.items() if not ok]
        
        # Check platform dependencies  
        platform_results = self.check_platform_dependencies()
        missing_platform = [mod for mod, (ok, _) in platform_results.items() if not ok]
        
        # Check system dependencies
        system_results = self.check_system_dependencies()
        missing_system = [pkg for pkg, (ok, _) in system_results.items() if not ok]
        
        # Display results
        all_missing = missing_core + missing_platform
        
        if not all_missing and not missing_system:
            print("\n✅ All dependencies are satisfied!")
            return True
        
        print(f"\n📊 Dependency Summary:")
        print(f"   Core dependencies missing: {len(missing_core)}")
        print(f"   Platform dependencies missing: {len(missing_platform)}")  
        print(f"   System dependencies missing: {len(missing_system)}")
        
        # Show missing dependencies
        if missing_core:
            print(f"\n❌ Missing core dependencies:")
            for mod in missing_core:
                print(f"   - {core_results[mod][1]}")
        
        if missing_platform:
            print(f"\n❌ Missing platform dependencies:")
            for mod in missing_platform:
                print(f"   - {platform_results[mod][1]}")
        
        if missing_system:
            print(f"\n⚠️  Missing system dependencies:")
            for pkg in missing_system:
                print(f"   - {system_results[pkg][1]}")
            print("   Note: System dependencies may require manual installation")
        
        # Auto-install if requested
        if auto_install and all_missing:
            print(f"\n🔧 Attempting to auto-install missing Python dependencies...")
            success = self.auto_install_missing(all_missing)
            
            if success:
                print(f"\n🔄 Re-checking dependencies after installation...")
                # Re-check only the ones we tried to install
                for dep in all_missing:
                    if self.check_import(dep):
                        version = self.get_package_version(dep)
                        print(f"✅ {dep} ({version})")
                    else:
                        print(f"❌ {dep} still missing")
                
                # Final verification
                final_missing = [mod for mod in all_missing if not self.check_import(mod)]
                if not final_missing:
                    print(f"\n🎉 All dependencies successfully installed!")
                    return True
                else:
                    print(f"\n⚠️  Some dependencies could not be installed automatically")
                    return False
            else:
                print(f"\n❌ Auto-installation failed")
                return False
        
        return not all_missing


def check_dependencies(project_root: Path, auto_install: bool = True) -> bool:
    """Convenience function to check dependencies"""
    checker = DependencyChecker(project_root)
    return checker.run_full_check(auto_install=auto_install)


if __name__ == "__main__":
    # Allow running as standalone script
    # Calculate project root relative to this file's location
    current_file = Path(__file__).resolve()
    # We are in src/ai_agent/utils/, so project root is three levels up
    project_root = current_file.parent.parent.parent
    success = check_dependencies(project_root)
    sys.exit(0 if success else 1)
