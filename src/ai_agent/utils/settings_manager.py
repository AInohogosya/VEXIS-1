"""
Settings Manager for VEXIS-1 AI Agent
Handles API key storage and model configuration
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

from ..utils.logger import get_logger


@dataclass
class APISettings:
    """API settings data structure"""
    google_api_key: Optional[str] = None
    preferred_provider: str = "ollama"  # "ollama" or "google"
    save_api_key: bool = True


class SettingsManager:
    """Manages application settings and API keys"""
    
    def __init__(self):
        self.logger = get_logger("settings_manager")
        self.settings_file = Path.cwd() / ".vexis" / "settings.json"
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        self._settings = self._load_settings()
    
    def _load_settings(self) -> APISettings:
        """Load settings from file"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    data = json.load(f)
                return APISettings(**data)
            else:
                return APISettings()
        except Exception as e:
            self.logger.warning(f"Failed to load settings: {e}")
            return APISettings()
    
    def _save_settings(self):
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(asdict(self._settings), f, indent=2)
            self.logger.info("Settings saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save settings: {e}")
    
    def get_settings(self) -> APISettings:
        """Get current settings"""
        return self._settings
    
    def set_google_api_key(self, api_key: str, save_key: bool = True):
        """Set Google API key"""
        self._settings.google_api_key = api_key
        self._settings.save_api_key = save_key
        self._save_settings()
        self.logger.info("Google API key updated")
    
    def set_preferred_provider(self, provider: str):
        """Set preferred provider"""
        if provider not in ["ollama", "google"]:
            raise ValueError("Provider must be 'ollama' or 'google'")
        self._settings.preferred_provider = provider
        self._save_settings()
        self.logger.info(f"Preferred provider set to: {provider}")
    
    def get_google_api_key(self) -> Optional[str]:
        """Get Google API key"""
        return self._settings.google_api_key
    
    def get_preferred_provider(self) -> str:
        """Get preferred provider"""
        return self._settings.preferred_provider
    
    def has_google_api_key(self) -> bool:
        """Check if Google API key is available"""
        return bool(self._settings.google_api_key)
    
    def clear_google_api_key(self):
        """Clear Google API key"""
        self._settings.google_api_key = None
        self._save_settings()
        self.logger.info("Google API key cleared")


# Global settings manager instance
_settings_manager: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    """Get global settings manager instance"""
    global _settings_manager
    
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    
    return _settings_manager
