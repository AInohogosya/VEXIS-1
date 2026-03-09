#!/usr/bin/env python3
"""
Dependency Installer
Handles package installation and dependency management
"""

import subprocess
import socket
from pathlib import Path
from typing import List

from .venv_manager import VirtualEnvManager


class DependencyInstaller:
    """Manages dependency installation"""
    
    def __init__(self, venv_manager: VirtualEnvManager) -> None:
        """Initialize dependency installer
        
        Args:
            venv_manager: Virtual environment manager instance
        """
        self.venv_manager = venv_manager
        self.max_retries = 3
    
    def check_network_connectivity(self) -> bool:
        """Check if network connectivity is available
        
        Returns:
            True if network connectivity to PyPI is available
        """
        try:
            socket.create_connection(("pypi.org", 443), timeout=10)
            return True
        except Exception:
            return False
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip with retry mechanism
        
        Returns:
            True if pip upgrade was successful
        """
        venv_python = self.venv_manager.get_venv_python_path()
        if not venv_python:
            return False
        
        for attempt in range(self.max_retries):
            try:
                result = subprocess.run(
                    [venv_python, "-m", "pip", "install", "--upgrade", "pip"],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    return True
            except (subprocess.TimeoutExpired, Exception):
                if attempt == self.max_retries - 1:
                    return False
        return False
    
    def install_requirements(self, requirements_files: List[Path]) -> bool:
        """Install from requirements files
        
        Args:
            requirements_files: List of requirements file paths
            
        Returns:
            True if installation was successful
        """
        venv_python = self.venv_manager.get_venv_python_path()
        if not venv_python:
            return False
        
        for requirements_file in requirements_files:
            if not requirements_file.exists():
                continue
            
            for attempt in range(self.max_retries):
                try:
                    result = subprocess.run(
                        [venv_python, "-m", "pip", "install", "-r", str(requirements_file)],
                        capture_output=True, text=True, timeout=600
                    )
                    if result.returncode == 0:
                        return True
                except (subprocess.TimeoutExpired, Exception):
                    if attempt == self.max_retries - 1:
                        return False
        return False
    
    def install_project(self, project_root: Path) -> bool:
        """Install project in editable mode
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            True if project installation was successful
        """
        venv_python = self.venv_manager.get_venv_python_path()
        if not venv_python:
            return False
        
        pyproject_file = project_root / "pyproject.toml"
        if not pyproject_file.exists():
            return True
        
        try:
            result = subprocess.run(
                [venv_python, "-m", "pip", "install", "-e", str(project_root)],
                capture_output=True, text=True, timeout=300
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
    
    def install_all_dependencies(self, project_root: Path) -> bool:
        """Install all dependencies with proper error handling
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            True if all dependencies were installed successfully
        """
        if not self.check_network_connectivity():
            print("Warning: No network connectivity")
        
        if not self.upgrade_pip():
            print("Warning: pip upgrade failed, continuing...")
        
        requirements_files = [
            project_root / "requirements-core.txt",
            project_root / "requirements.txt"
        ]
        
        if not self.install_requirements(requirements_files):
            return False
        
        if not self.install_project(project_root):
            print("Warning: Project installation failed")
        
        return True
    
    def request_dependency_installation_permission(self, project_root: Path) -> bool:
        """Ask user for permission to install dependencies
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            True if user grants permission
        """
        print("\n" + "="*60)
        print("📦 Dependency Installation Required")
        print("="*60)
        print("This program needs to install dependencies to:")
        print("  • Provide AI model integration")
        print("  • Enable GUI automation capabilities")
        print("  • Support cross-platform functionality")
        print()
        
        # Show main dependencies that will be installed
        pyproject_file = project_root / "pyproject.toml"
        if pyproject_file.exists():
            try:
                with open(pyproject_file, 'r') as f:
                    content = f.read()
                    # Extract some key dependencies for display
                    import re
                    deps_match = re.search(r'dependencies = \[(.*?)\]', content, re.DOTALL)
                    if deps_match:
                        deps_text = deps_match.group(1)
                        # Extract first few important packages
                        important_packages = []
                        for line in deps_text.split('\n'):
                            line = line.strip().strip('"').strip("'")
                            if line and not line.startswith('#'):
                                pkg_name = line.split('>=')[0].split('==')[0]
                                if pkg_name in ['Pillow', 'pyautogui', 'requests', 'opencv-python', 
                                              'numpy', 'openai', 'anthropic', 'transformers', 'torch']:
                                    important_packages.append(pkg_name)
                                    if len(important_packages) >= 5:
                                        break
                        
                        if important_packages:
                            print("Key packages to install:")
                            for pkg in important_packages:
                                print(f"  • {pkg}")
                            print(f"  • ... and {len(deps_text.split()) - len(important_packages)} more packages")
            except Exception:
                pass
        
        print()
        print("Dependencies will be installed inside the virtual environment.")
        print("This will not affect your system Python installation.")
        print()
        
        while True:
            response = input("Is it okay to install the dependencies inside the virtual environment? (y/n): ").strip().lower()
            if response in ['y', 'yes', 'yeah', 'yep']:
                return True
            elif response in ['n', 'no', 'nope']:
                print("Dependency installation declined.")
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    
    def install_all_dependencies_with_consent(self, project_root: Path) -> bool:
        """Install all dependencies with user consent
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            True if all dependencies were installed successfully or user declined
        """
        if not self.request_dependency_installation_permission(project_root):
            return False
        
        print("Installing dependencies...")
        
        if not self.check_network_connectivity():
            print("Warning: No network connectivity")
        
        if not self.upgrade_pip():
            print("Warning: pip upgrade failed, continuing...")
        
        requirements_files = [
            project_root / "requirements-core.txt",
            project_root / "requirements.txt"
        ]
        
        if not self.install_requirements(requirements_files):
            print("✗ Failed to install requirements")
            return False
        
        if not self.install_project(project_root):
            print("Warning: Project installation failed")
        
        print("✓ Dependencies installed successfully")
        return True
