#!/usr/bin/env python3
"""
Virtual Environment Manager
Handles virtual environment creation, detection, and management
"""

import sys
import os
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Optional

VENV_DIR = "venv"


class VirtualEnvManager:
    """Manages virtual environment operations"""
    
    def __init__(self, project_root: Path) -> None:
        """Initialize virtual environment manager
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.venv_path = project_root / VENV_DIR
    
    def is_in_virtual_environment(self) -> bool:
        """Check if currently running in a virtual environment
        
        Returns:
            True if running in a virtual environment
        """
        return (
            hasattr(sys, 'real_prefix') or 
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            os.getenv('VIRTUAL_ENV') is not None
        )
    
    def get_venv_python_path(self) -> Optional[str]:
        """Get the Python executable path in the virtual environment
        
        Returns:
            Path to Python executable in venv, or None if not found
        """
        if not self.venv_path.exists():
            return None
        
        if platform.system() == "Windows":
            python_exe = self.venv_path / "Scripts" / "python.exe"
            if not python_exe.exists():
                python_exe = self.venv_path / "Scripts" / "pythonw.exe"
        else:
            python_exe = self.venv_path / "bin" / "python"
            if not python_exe.exists():
                python_exe = self.venv_path / "bin" / "python3"
        
        return str(python_exe) if python_exe.exists() else None
    
    def check_prerequisites(self) -> bool:
        """Check if virtual environment creation prerequisites are met
        
        Returns:
            True if venv module is available
        """
        try:
            import venv
            return True
        except ImportError:
            return False
    
    def create_environment(self) -> bool:
        """Create a virtual environment with error handling
        
        Returns:
            True if environment was created successfully
        """
        if self.venv_path.exists():
            venv_python = self.get_venv_python_path()
            if venv_python:
                try:
                    result = subprocess.run([venv_python, "--version"], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        return True
                except Exception:
                    pass
                shutil.rmtree(self.venv_path)
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "venv", str(self.venv_path)],
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
    
    def request_venv_creation_permission(self) -> bool:
        """Ask user for permission to create virtual environment
        
        Returns:
            True if user grants permission
        """
        print("\n" + "="*60)
        print("🔧 Virtual Environment Setup Required")
        print("="*60)
        print("This program needs to create a virtual environment to:")
        print("  • Isolate dependencies from your system Python")
        print("  • Prevent package conflicts")
        print("  • Ensure reproducible behavior")
        print()
        print(f"Location: {self.venv_path}")
        print()
        
        while True:
            response = input("Is it okay to create the virtual environment? (y/n): ").strip().lower()
            if response in ['y', 'yes', 'yeah', 'yep']:
                return True
            elif response in ['n', 'no', 'nope']:
                print("Virtual environment creation declined.")
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    
    def create_environment_with_consent(self) -> bool:
        """Create virtual environment with user consent
        
        Returns:
            True if environment was created successfully or user declined
        """
        if self.venv_path.exists():
            venv_python = self.get_venv_python_path()
            if venv_python:
                try:
                    result = subprocess.run([venv_python, "--version"], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        print("✓ Virtual environment already exists")
                        return True
                except Exception:
                    pass
        
        if not self.request_venv_creation_permission():
            return False
        
        print("Creating virtual environment...")
        if self.create_environment():
            print("✓ Virtual environment created successfully")
            return True
        else:
            print("✗ Failed to create virtual environment")
            return False
    
    def restart_in_venv(self, args: list[str], script_path: str = None) -> bool:
        """Restart the current script in the virtual environment
        
        Args:
            args: Command line arguments to pass to the restarted script
            script_path: Path to the script to restart (defaults to run.py in project root)
            
        Returns:
            True if restart was initiated successfully
        """
        venv_python = self.get_venv_python_path()
        if not venv_python:
            return False
        
        if script_path is None:
            script_path = str(self.project_root / "run.py")
        
        new_argv = [venv_python, script_path, "--__venv_restarted__"] + args
        
        try:
            os.execv(venv_python, new_argv)
            return True
        except OSError:
            return False
