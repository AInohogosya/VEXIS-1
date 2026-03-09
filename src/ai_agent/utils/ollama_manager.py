#!/usr/bin/env python3
"""
Ollama Model Manager for VEXIS-1.2
Handles model validation, installation, and management
Optimized for Gemini family models
"""

import subprocess
import json
import time
from typing import Optional, List, Dict, Any
from ..utils.logger import get_logger
from .model_definitions import (
    PREDEFINED_GEMINI_MODELS, get_gemini_model_info, 
    is_gemini_model, get_local_gemini_models, get_cloud_gemini_models
)


class OllamaManager:
    """Manages Ollama models with validation and installation"""
    
    def __init__(self):
        self.logger = get_logger("ollama_manager")
        self.endpoint = "http://localhost:11434"
    
    def check_ollama_available(self) -> bool:
        """Check if Ollama is available and running"""
        try:
            response = subprocess.run(["ollama", "--version"], 
                                    capture_output=True, text=True, timeout=10)
            return response.returncode == 0
        except Exception:
            return False
    
    def get_installed_models(self) -> List[str]:
        """Get list of installed models"""
        try:
            response = subprocess.run(["ollama", "list"], 
                                    capture_output=True, text=True, timeout=30)
            if response.returncode == 0:
                lines = response.stdout.strip().split('\n')
                models = []
                for line in lines[1:]:  # Skip header line
                    if line.strip():
                        # Extract model name from the first column
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            else:
                self.logger.error(f"Failed to list models: {response.stderr}")
                return []
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []
    
    def is_model_installed(self, model_name: str) -> bool:
        """Check if a specific model is installed"""
        installed_models = self.get_installed_models()
        return model_name in installed_models
    
    def validate_gemini_model(self, model_name: str) -> Dict[str, Any]:
        """Validate if a Gemini model name exists and is available"""
        result = {
            "valid": False,
            "installed": False,
            "gemini_model": False,
            "type": None,
            "error": None,
            "model_name": model_name,
            "model_info": {}
        }
        
        # Check if it's a Gemini model
        if is_gemini_model(model_name):
            result["gemini_model"] = True
            result["type"] = PREDEFINED_GEMINI_MODELS[model_name]["type"]
            result["model_info"] = get_gemini_model_info(model_name)
        
        # Check if model is installed
        installed = self.is_model_installed(model_name)
        result["installed"] = installed
        
        if installed:
            result["valid"] = True
        elif is_gemini_model(model_name):
            # Gemini models are considered valid even if not installed
            result["valid"] = True
        else:
            # For custom models, try to validate by checking if it exists in Ollama
            try:
                response = subprocess.run(["ollama", "show", model_name], 
                                        capture_output=True, text=True, timeout=30)
                if response.returncode == 0:
                    result["valid"] = True
                    result["type"] = "cloud" if "-cloud" in model_name else "local"
                else:
                    result["error"] = f"Model '{model_name}' not found in Ollama"
            except Exception as e:
                result["error"] = f"Error validating model: {e}"
        
        return result
    
    def install_model(self, model_name: str) -> Dict[str, Any]:
        """Install a model if not already installed"""
        result = {
            "success": False,
            "installed": False,
            "error": None,
            "model_name": model_name
        }
        
        if self.is_model_installed(model_name):
            result["installed"] = True
            result["success"] = True
            return result
        
        try:
            self.logger.info(f"Installing model: {model_name}")
            
            # Show progress for large models
            if is_gemini_model(model_name):
                model_info = get_gemini_model_info(model_name)
                print(f"📥 Installing {model_info.get('name', model_name)}")
                print(f"   {model_info.get('desc', 'Installing model...')}")
            
            response = subprocess.run(["ollama", "pull", model_name], 
                                   capture_output=True, text=True, timeout=600)  # 10 minutes timeout
            
            if response.returncode == 0:
                result["installed"] = True
                result["success"] = True
                self.logger.info(f"Successfully installed model: {model_name}")
            else:
                result["error"] = f"Failed to install model: {response.stderr}"
                self.logger.error(f"Failed to install {model_name}: {response.stderr}")
                
        except subprocess.TimeoutExpired:
            result["error"] = f"Installation timeout for model: {model_name}"
            self.logger.error(f"Installation timeout for {model_name}")
        except Exception as e:
            result["error"] = f"Error installing model: {e}"
            self.logger.error(f"Error installing {model_name}: {e}")
        
        return result
    
    def remove_model(self, model_name: str) -> Dict[str, Any]:
        """Remove a model"""
        result = {
            "success": False,
            "error": None,
            "model_name": model_name
        }
        
        if not self.is_model_installed(model_name):
            result["success"] = True  # Already removed
            return result
        
        try:
            self.logger.info(f"Removing model: {model_name}")
            response = subprocess.run(["ollama", "rm", model_name], 
                                   capture_output=True, text=True, timeout=120)
            
            if response.returncode == 0:
                result["success"] = True
                self.logger.info(f"Successfully removed model: {model_name}")
            else:
                result["error"] = f"Failed to remove model: {response.stderr}"
                self.logger.error(f"Failed to remove {model_name}: {response.stderr}")
                
        except Exception as e:
            result["error"] = f"Error removing model: {e}"
            self.logger.error(f"Error removing {model_name}: {e}")
        
        return result
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        result = {
            "success": False,
            "info": {},
            "error": None,
            "model_name": model_name
        }
        
        try:
            response = subprocess.run(["ollama", "show", model_name], 
                                   capture_output=True, text=True, timeout=30)
            
            if response.returncode == 0:
                # Parse the output to extract model information
                info = {}
                for line in response.stdout.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        info[key.strip()] = value.strip()
                
                result["info"] = info
                result["success"] = True
            else:
                result["error"] = f"Failed to get model info: {response.stderr}"
                
        except Exception as e:
            result["error"] = f"Error getting model info: {e}"
        
        return result
    
    def get_recommended_models(self) -> List[Dict[str, Any]]:
        """Get recommended Gemini models"""
        recommended = []
        
        # Add lightweight option
        lightweight_models = [m for m in get_local_gemini_models() if any(s in m for s in [":1b", ":2b"])]
        if lightweight_models:
            model = lightweight_models[0]
            info = get_gemini_model_info(model)
            recommended.append({
                "model": model,
                "name": info.get("name", model),
                "desc": info.get("desc", ""),
                "reason": "Lightweight and efficient",
                "category": "lightweight"
            })
        
        # Add balanced option
        if "gemma2:2b" in get_local_gemini_models():
            info = get_gemini_model_info("gemma2:2b")
            recommended.append({
                "model": "gemma2:2b",
                "name": info.get("name", "Gemma 2 2B"),
                "desc": info.get("desc", ""),
                "reason": "Good balance of performance and efficiency",
                "category": "balanced"
            })
        
        # Add cloud option
        cloud_models = get_cloud_gemini_models()
        if cloud_models:
            model = cloud_models[0]
            info = get_gemini_model_info(model)
            recommended.append({
                "model": model,
                "name": info.get("name", model),
                "desc": info.get("desc", ""),
                "reason": "Cloud-based with latest features",
                "category": "cloud"
            })
        
        return recommended
    
    def check_cloud_model_access(self) -> Dict[str, Any]:
        """Check if cloud models are accessible"""
        result = {
            "available": False,
            "signed_in": False,
            "version_ok": False,
            "error": None
        }
        
        # Check Ollama version
        try:
            version_response = subprocess.run(["ollama", "--version"], 
                                          capture_output=True, text=True, timeout=10)
            if version_response.returncode == 0:
                # Extract version number
                import re
                version_match = re.search(r"version is (\d+\.\d+\.\d+)", version_response.stdout)
                if version_match:
                    version = version_match.group(1)
                    major, minor, patch = map(int, version.split('.'))
                    result["version_ok"] = major >= 1 or (major == 0 and minor >= 17)
        except Exception:
            pass
        
        # Check if signed in
        try:
            whoami_response = subprocess.run(["ollama", "whoami"], 
                                          capture_output=True, text=True, timeout=10)
            if whoami_response.returncode == 0 and "not signed in" not in whoami_response.stderr.lower():
                result["signed_in"] = True
        except Exception:
            pass
        
        result["available"] = result["version_ok"] and result["signed_in"]
        
        return result
    
    def list_gemini_models(self, installed_only: bool = False) -> List[Dict[str, Any]]:
        """List only Gemini 3.1 Pro and Gemini 3 Flash models"""
        models = []
        
        # Only show two requested models
        allowed_models = ["gemini-3-pro-preview", "gemini-3-flash-preview"]
        
        if installed_only:
            installed_models = self.get_installed_models()
            gemini_models = [m for m in installed_models if m in allowed_models]
        else:
            gemini_models = allowed_models
        
        for model_name in gemini_models:
            validation = self.validate_gemini_model(model_name)
            model_info = get_gemini_model_info(model_name)
            
            models.append({
                "name": model_name,
                "display_name": model_info.get("name", model_name),
                "description": model_info.get("desc", ""),
                "icon": model_info.get("icon", "📋"),
                "installed": validation["installed"],
                "type": validation["type"],
                "valid": validation["valid"],
                "error": validation.get("error")
            })
        
        return sorted(models, key=lambda x: (not x["installed"], x["name"]))


if __name__ == "__main__":
    # Test the Ollama manager
    manager = OllamaManager()
    
    print("Ollama available:", manager.check_ollama_available())
    print("\nInstalled models:")
    for model in manager.get_installed_models():
        print(f"- {model}")
    
    print("\nGemini models:")
    for model in manager.list_gemini_models():
        status = "✓" if model["installed"] else "○"
        print(f"{status} {model['icon']} {model['display_name']} - {model['description']}")
    
    print("\nRecommended models:")
    for rec in manager.get_recommended_models():
        print(f"- {rec['name']}: {rec['reason']}")
