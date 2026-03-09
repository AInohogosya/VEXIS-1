#!/usr/bin/env python3
"""
Comprehensive System Check for VEXIS-1.2
Validates all components and provides detailed diagnostics
"""

import subprocess
import sys
import os
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


class SystemChecker:
    """Comprehensive system validation"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.issues = []
        self.warnings = []
        self.successes = []
    
    def log(self, status: str, message: str, details: str = ""):
        """Log a check result"""
        if status == "success":
            self.successes.append(f"✅ {message}")
            if details:
                self.successes.append(f"   {details}")
        elif status == "warning":
            self.warnings.append(f"⚠️  {message}")
            if details:
                self.warnings.append(f"   {details}")
        elif status == "error":
            self.issues.append(f"❌ {message}")
            if details:
                self.issues.append(f"   {details}")
    
    def check_python_version(self):
        """Check Python version compatibility"""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            self.log("success", f"Python {version.major}.{version.minor}.{version.micro} is compatible")
        else:
            self.log("error", f"Python {version.major}.{version.minor}.{version.micro} is too old", 
                     "Requires Python 3.8+")
    
    def check_virtual_environment(self):
        """Check if running in virtual environment"""
        if sys.prefix != sys.base_prefix:
            self.log("success", "Running in virtual environment", f"Path: {sys.prefix}")
        else:
            self.log("warning", "Not running in virtual environment", 
                     "Recommended: Create and activate a virtual environment")
    
    def check_required_modules(self):
        """Check if required modules can be imported"""
        required_modules = [
            ("json", "Standard library"),
            ("pathlib", "Standard library"),
            ("subprocess", "Standard library"),
            ("typing", "Standard library"),
            ("dataclasses", "Standard library"),
            ("enum", "Standard library"),
        ]
        
        optional_modules = [
            ("curses", "Terminal UI"),
            ("requests", "HTTP requests"),
            ("yaml", "YAML configuration"),
        ]
        
        for module_name, description in required_modules:
            try:
                importlib.util.find_spec(module_name)
                self.log("success", f"Module {module_name} available", description)
            except ImportError:
                self.log("error", f"Required module {module_name} not found", description)
        
        for module_name, description in optional_modules:
            try:
                importlib.util.find_spec(module_name)
                self.log("success", f"Optional module {module_name} available", description)
            except ImportError:
                self.log("warning", f"Optional module {module_name} not found", 
                         f"Install with: pip install {module_name}")
    
    def check_project_structure(self):
        """Check project directory structure"""
        required_dirs = [
            "src",
            "src/ai_agent",
            "src/ai_agent/utils",
            "src/ai_agent/core_processing",
            "src/ai_agent/external_integration",
            "src/ai_agent/platform_abstraction",
            "src/ai_agent/user_interface",
        ]
        
        required_files = [
            "run.py",
            "src/ai_agent/__init__.py",
            "src/ai_agent/utils/__init__.py",
            "src/ai_agent/core_processing/__init__.py",
            "src/ai_agent/external_integration/__init__.py",
            "src/ai_agent/platform_abstraction/__init__.py",
            "src/ai_agent/user_interface/__init__.py",
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                self.log("success", f"Directory {dir_path} exists")
            else:
                self.log("error", f"Required directory {dir_path} missing")
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists() and full_path.is_file():
                self.log("success", f"File {file_path} exists")
            else:
                self.log("error", f"Required file {file_path} missing")
    
    def check_ollama_installation(self):
        """Check Ollama installation and status"""
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.log("success", "Ollama is installed", f"Version: {version}")
                
                # Check if version supports cloud models
                if "0.17" in version or any(v in version for v in ["0.18", "0.19", "1."]):
                    self.log("success", "Ollama version supports cloud models")
                else:
                    self.log("warning", "Ollama version may not support cloud models", 
                             "Update with: curl -fsSL https://ollama.com/install.sh | sh")
            else:
                self.log("error", "Ollama command failed", result.stderr)
        except FileNotFoundError:
            self.log("warning", "Ollama not found", 
                     "Install with: curl -fsSL https://ollama.com/install.sh | sh")
        except Exception as e:
            self.log("error", "Error checking Ollama", str(e))
    
    def check_ollama_service(self):
        """Check if Ollama service is running"""
        try:
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.log("success", "Ollama service is running")
                
                # Count models
                lines = result.stdout.strip().split('\n')
                model_count = len([line for line in lines[1:] if line.strip()])
                self.log("success", f"Ollama has {model_count} models installed")
            else:
                self.log("error", "Ollama service not responding", result.stderr)
        except Exception as e:
            self.log("error", "Error checking Ollama service", str(e))
    
    def check_gemini_models(self):
        """Check Gemini model availability"""
        try:
            # Import here to avoid import errors if modules are missing
            from ai_agent.utils.model_definitions import get_local_gemini_models, get_cloud_gemini_models
            
            local_models = get_local_gemini_models()
            cloud_models = get_cloud_gemini_models()
            
            if local_models:
                self.log("success", f"Found {len(local_models)} local Gemini models")
                for model in local_models[:3]:  # Show first 3
                    self.log("success", f"  Available: {model}")
            else:
                self.log("warning", "No local Gemini models found", 
                         "Install with: ollama pull gemma2:2b")
            
            if cloud_models:
                self.log("success", f"Found {len(cloud_models)} cloud Gemini models")
            else:
                self.log("info", "No cloud Gemini models configured")
                
        except ImportError as e:
            self.log("warning", "Could not check Gemini models", str(e))
    
    def check_network_connectivity(self):
        """Check network connectivity"""
        test_urls = [
            ("ollama.com", "Ollama registry"),
            ("pypi.org", "Python package index"),
            ("google.com", "General internet"),
        ]
        
        for host, description in test_urls:
            try:
                import socket
                socket.create_connection((host, 443), timeout=5)
                self.log("success", f"Can connect to {host}", description)
            except Exception:
                self.log("warning", f"Cannot connect to {host}", 
                         f"May affect {description} access")
    
    def check_file_permissions(self):
        """Check file permissions for key directories"""
        key_dirs = [
            self.project_root,
            self.project_root / "src",
            Path.home() / ".vexis",
        ]
        
        for dir_path in key_dirs:
            try:
                if dir_path.exists():
                    # Test write permission
                    test_file = dir_path / ".vexis_test_write"
                    test_file.touch()
                    test_file.unlink()
                    self.log("success", f"Write permission OK for {dir_path}")
                else:
                    # Try to create directory
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.log("success", f"Created directory {dir_path}")
            except Exception as e:
                self.log("error", f"Permission issue with {dir_path}", str(e))
    
    def check_dependencies_file(self):
        """Check for requirements files"""
        req_files = [
            "requirements.txt",
            "requirements-core.txt",
            "pyproject.toml",
        ]
        
        for req_file in req_files:
            file_path = self.project_root / req_file
            if file_path.exists():
                self.log("success", f"Found {req_file}")
            else:
                self.log("info", f"Optional file {req_file} not found")
    
    def run_all_checks(self) -> Dict[str, int]:
        """Run all system checks"""
        print("🔍 Running comprehensive system check...")
        print("=" * 60)
        
        # Run all checks
        self.check_python_version()
        self.check_virtual_environment()
        self.check_required_modules()
        self.check_project_structure()
        self.check_ollama_installation()
        self.check_ollama_service()
        self.check_gemini_models()
        self.check_network_connectivity()
        self.check_file_permissions()
        self.check_dependencies_file()
        
        # Return summary
        return {
            "successes": len(self.successes),
            "warnings": len(self.warnings),
            "errors": len(self.issues)
        }
    
    def display_results(self):
        """Display check results"""
        print("\n" + "=" * 60)
        print("📊 SYSTEM CHECK RESULTS")
        print("=" * 60)
        
        if self.successes:
            print(f"\n✅ SUCCESSES ({len(self.successes)}):")
            for success in self.successes:
                print(f"  {success}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if self.issues:
            print(f"\n❌ ERRORS ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  {issue}")
        
        # Summary
        total = len(self.successes) + len(self.warnings) + len(self.issues)
        success_rate = (len(self.successes) / total * 100) if total > 0 else 0
        
        print(f"\n📈 SUMMARY:")
        print(f"  Total checks: {total}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        if len(self.issues) == 0:
            print(f"  🎉 System is ready for VEXIS-1.2!")
        else:
            print(f"  🔧 Fix {len(self.issues)} issues before running VEXIS-1.2")
        
        print("=" * 60)
        
        return len(self.issues) == 0


def main():
    verbose = "--verbose" in sys.argv
    fix_mode = "--fix" in sys.argv
    
    checker = SystemChecker()
    summary = checker.run_all_checks()
    
    if verbose:
        checker.display_results()
    else:
        # Brief summary
        print(f"\n📊 Summary: {summary['successes']} successes, {summary['warnings']} warnings, {summary['errors']} errors")
        
        if summary['errors'] > 0:
            print("❌ System check failed. Run with --verbose for details.")
            sys.exit(1)
        elif summary['warnings'] > 0:
            print("⚠️  System check passed with warnings. Run with --verbose for details.")
        else:
            print("✅ System check passed successfully!")
    
    # Auto-fix if requested
    if fix_mode and summary['errors'] > 0:
        print("\n🔧 Auto-fix mode not implemented yet.")
        print("Please fix the issues manually or run: python3 check_environment.py --fix")


if __name__ == "__main__":
    main()
