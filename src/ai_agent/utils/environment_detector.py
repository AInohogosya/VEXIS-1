#!/usr/bin/env python3
"""
Environment Detection and Adaptive Execution System for VEXIS-1.2
Gathers system data and adapts execution based on the environment
Optimized for VEXIS-1.2 architecture
"""

import subprocess
import platform
import json
import os
import sys
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class EnvironmentInfo:
    """Comprehensive environment information"""
    # OS Information
    os_system: str
    os_release: str
    os_version: str
    os_machine: str
    
    # Python Information
    python_version: str
    python_executable: str
    
    # Ollama Information
    ollama_available: bool
    ollama_version: Optional[str]
    ollama_has_signin: bool
    ollama_has_whoami: bool
    ollama_installed_models: List[str]
    ollama_cloud_models: List[str]
    ollama_local_models: List[str]
    
    # System Capabilities
    has_venv_module: bool
    has_docker: bool
    has_git: bool
    has_curl: bool
    
    # Network Status
    can_connect_to_ollama_com: bool
    can_connect_to_pypi: bool
    
    # Recommendations
    needs_ollama_update: bool
    can_use_cloud_models: bool
    recommended_provider: str
    warnings: List[str]
    suggestions: List[str]


class EnvironmentDetector:
    """Detects and analyzes the runtime environment"""
    
    def __init__(self):
        self.warnings: List[str] = []
        self.suggestions: List[str] = []
    
    def detect_all(self) -> EnvironmentInfo:
        """Run all detection commands and return comprehensive info"""
        return EnvironmentInfo(
            # OS Information
            os_system=self._detect_os_system(),
            os_release=self._detect_os_release(),
            os_version=self._detect_os_version(),
            os_machine=self._detect_os_machine(),
            
            # Python Information
            python_version=self._detect_python_version(),
            python_executable=self._detect_python_executable(),
            
            # Ollama Information
            ollama_available=self._detect_ollama_available(),
            ollama_version=self._detect_ollama_version(),
            ollama_has_signin=self._detect_ollama_has_signin(),
            ollama_has_whoami=self._detect_ollama_has_whoami(),
            ollama_installed_models=self._detect_ollama_models(),
            ollama_cloud_models=self._detect_cloud_models(),
            ollama_local_models=self._detect_local_models(),
            
            # System Capabilities
            has_venv_module=self._detect_venv_module(),
            has_docker=self._detect_docker(),
            has_git=self._detect_git(),
            has_curl=self._detect_curl(),
            
            # Network Status
            can_connect_to_ollama_com=self._detect_ollama_com_connectivity(),
            can_connect_to_pypi=self._detect_pypi_connectivity(),
            
            # Recommendations
            needs_ollama_update=self._detect_needs_ollama_update(),
            can_use_cloud_models=self._detect_can_use_cloud_models(),
            recommended_provider=self._detect_recommended_provider(),
            warnings=self.warnings,
            suggestions=self.suggestions
        )
    
    def _detect_os_system(self) -> str:
        """Detect OS system name"""
        return platform.system()
    
    def _detect_os_release(self) -> str:
        """Detect OS release"""
        try:
            if platform.system() == "Linux":
                with open("/etc/os-release", "r") as f:
                    for line in f:
                        if line.startswith("ID="):
                            return line.strip().split("=")[1].strip('"')
            return platform.release()
        except Exception:
            return "Unknown"
    
    def _detect_os_version(self) -> str:
        """Detect OS version"""
        return platform.version()
    
    def _detect_os_machine(self) -> str:
        """Detect machine architecture"""
        return platform.machine()
    
    def _detect_python_version(self) -> str:
        """Detect Python version"""
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _detect_python_executable(self) -> str:
        """Detect Python executable path"""
        return sys.executable
    
    def _detect_ollama_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False
    
    def _detect_ollama_version(self) -> Optional[str]:
        """Get Ollama version"""
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_match = re.search(r"version is (\d+\.\d+\.\d+)", result.stdout)
                if version_match:
                    return version_match.group(1)
            return None
        except Exception:
            return None
    
    def _detect_ollama_has_signin(self) -> bool:
        """Check if Ollama has signin command"""
        try:
            result = subprocess.run(["ollama", "--help"], 
                                  capture_output=True, text=True, timeout=10)
            return "signin" in result.stdout
        except Exception:
            return False
    
    def _detect_ollama_has_whoami(self) -> bool:
        """Check if Ollama has whoami command"""
        try:
            result = subprocess.run(["ollama", "--help"], 
                                  capture_output=True, text=True, timeout=10)
            return "whoami" in result.stdout
        except Exception:
            return False
    
    def _detect_ollama_models(self) -> List[str]:
        """Get list of installed Ollama models"""
        try:
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                models = []
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            return []
        except Exception:
            return []
    
    def _detect_cloud_models(self) -> List[str]:
        """Detect cloud models (models with -cloud suffix)"""
        models = self._detect_ollama_models()
        return [model for model in models if "-cloud" in model]
    
    def _detect_local_models(self) -> List[str]:
        """Detect local models (models without -cloud suffix)"""
        models = self._detect_ollama_models()
        return [model for model in models if "-cloud" not in model]
    
    def _detect_venv_module(self) -> bool:
        """Check if venv module is available"""
        try:
            import venv
            return True
        except ImportError:
            return False
    
    def _detect_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False
    
    def _detect_git(self) -> bool:
        """Check if Git is available"""
        try:
            result = subprocess.run(["git", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False
    
    def _detect_curl(self) -> bool:
        """Check if curl is available"""
        try:
            result = subprocess.run(["curl", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False
    
    def _detect_ollama_com_connectivity(self) -> bool:
        """Check if can connect to ollama.com"""
        try:
            import socket
            socket.create_connection(("ollama.com", 443), timeout=10)
            return True
        except Exception:
            return False
    
    def _detect_pypi_connectivity(self) -> bool:
        """Check if can connect to PyPI"""
        try:
            import socket
            socket.create_connection(("pypi.org", 443), timeout=10)
            return True
        except Exception:
            return False
    
    def _detect_needs_ollama_update(self) -> bool:
        """Check if Ollama needs update for cloud models"""
        if not self._detect_ollama_available():
            return False
        
        version = self._detect_ollama_version()
        if not version:
            return True
        
        try:
            major, minor, patch = map(int, version.split('.'))
            # Cloud models require 0.17.0+
            return major < 1 and minor < 17
        except Exception:
            return True
    
    def _detect_can_use_cloud_models(self) -> bool:
        """Check if cloud models can be used"""
        if not self._detect_ollama_available():
            return False
        
        if self._detect_needs_ollama_update():
            return False
        
        if not self._detect_ollama_has_whoami():
            return False
        
        # Check if signed in
        try:
            result = subprocess.run(["ollama", "whoami"], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0 and "not signed in" not in result.stderr.lower()
        except Exception:
            return False
    
    def _detect_recommended_provider(self) -> str:
        """Get recommended provider based on environment"""
        if self._detect_can_use_cloud_models():
            return "ollama"
        elif self._detect_ollama_available():
            return "ollama"
        else:
            return "google"


class AdaptiveExecutor:
    """Executes adaptive setup based on environment detection"""
    
    def __init__(self, env_info: EnvironmentInfo):
        self.env_info = env_info
        self.execution_plan: List[Dict[str, Any]] = []
        self._create_execution_plan()
    
    def _create_execution_plan(self):
        """Create execution plan based on environment"""
        
        # Ollama installation
        if not self.env_info.ollama_available:
            self.execution_plan.append({
                "action": "install_ollama",
                "description": "Install Ollama for local AI models",
                "priority": "high",
                "command": "curl -fsSL https://ollama.com/install.sh | sh"
            })
        
        # Ollama update
        elif self.env_info.needs_ollama_update:
            self.execution_plan.append({
                "action": "update_ollama",
                "description": "Update Ollama for cloud model support",
                "priority": "medium",
                "command": "curl -fsSL https://ollama.com/install.sh | sh"
            })
        
        # Ollama signin
        if self.env_info.ollama_available and self.env_info.ollama_has_whoami:
            try:
                result = subprocess.run(["ollama", "whoami"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode != 0 or "not signed in" in result.stderr.lower():
                    self.execution_plan.append({
                        "action": "ollama_signin",
                        "description": "Sign in to Ollama for cloud models",
                        "priority": "medium",
                        "command": "ollama signin"
                    })
            except Exception:
                pass
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations based on environment"""
        recommendations = []
        
        if not self.env_info.ollama_available:
            recommendations.append("Install Ollama for local AI model support")
        
        if self.env_info.needs_ollama_update:
            recommendations.append("Update Ollama to enable cloud model features")
        
        if not self.env_info.can_connect_to_pypi:
            recommendations.append("Check internet connection for package installation")
        
        if not self.env_info.has_venv_module:
            recommendations.append("Install python3-venv package for virtual environment support")
        
        return recommendations
    
    def execute_plan(self, interactive: bool = False) -> bool:
        """Execute the adaptive plan"""
        success = True
        
        for step in self.execution_plan:
            if interactive:
                print(f"\n🔧 {step['description']}")
                response = input("Execute this step? (y/n): ").lower().strip()
                if response != 'y':
                    continue
            
            try:
                print(f"Executing: {step['command']}")
                result = subprocess.run(step['command'], shell=True, 
                                      capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"✓ {step['description']} completed")
                else:
                    print(f"✗ {step['description']} failed")
                    print(f"Error: {result.stderr}")
                    success = False
                    
            except Exception as e:
                print(f"✗ {step['description']} failed with exception: {e}")
                success = False
        
        return success


def detect_and_plan() -> Tuple[EnvironmentInfo, AdaptiveExecutor]:
    """Convenience function to detect environment and create plan"""
    detector = EnvironmentDetector()
    env_info = detector.detect_all()
    executor = AdaptiveExecutor(env_info)
    return env_info, executor


if __name__ == "__main__":
    # Test the environment detector
    env_info, executor = detect_and_plan()
    print(json.dumps(asdict(env_info), indent=2))
    print("\nExecution Plan:")
    for step in executor.execution_plan:
        print(f"- {step['description']}")
