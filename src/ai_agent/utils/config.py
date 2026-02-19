"""
Configuration management for AI Agent System
Zero-defect policy: comprehensive configuration with validation
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from .exceptions import ConfigurationError, ValidationError


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    file: Optional[str] = None
    json_format: bool = False
    console: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class APIConfig:
    """API configuration"""
    # Local Ollama configuration
    local_endpoint: str = "http://localhost:11434"
    local_model: str = "gemini-3-flash-preview:latest"
    
    
    # General settings
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    preferred_provider: str = "ollama"  # "ollama" only


@dataclass
class GUIConfig:
    """GUI automation configuration"""
    click_delay: float = 0.1
    typing_delay: float = 0.05
    scroll_duration: float = 0.5
    drag_duration: float = 0.3
    screenshot_quality: int = 95
    screenshot_format: str = "PNG"


@dataclass
class SecurityConfig:
    """Security configuration"""
    max_coordinate_value: float = 1.0
    min_coordinate_value: float = 0.0
    coordinate_validation: bool = True
    allowed_commands: list = field(default_factory=lambda: [
        "click", "double_click", "right_click", "text", "key", "drag", "scroll", "end"
    ])
    sanitize_text_input: bool = True
    validate_file_paths: bool = True
    max_text_length: int = 1000


@dataclass
class PerformanceConfig:
    """Performance configuration"""
    max_concurrent_tasks: int = 1
    task_timeout: int = 0  # No task timeout
    screenshot_timeout: int = 10
    api_timeout: int = 30
    memory_limit_mb: int = 1024


@dataclass
class VerificationConfig:
    """Task completion verification configuration"""
    enabled: bool = True
    confidence_threshold: float = 0.8
    max_verification_attempts: int = 3
    max_regenerations: int = 2
    verification_model: str = "gemini-3-flash-preview:latest"
    verification_timeout: int = 60
    auto_regenerate: bool = True


@dataclass
class Config:
    """Main configuration class"""
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    
    # Platform-specific settings
    platform: Dict[str, Any] = field(default_factory=dict)
    
    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        keys = key.split('.')
        value = self
        
        try:
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except (AttributeError, KeyError):
            return default


class ConfigManager:
    """Configuration manager with validation and environment support"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self._config: Optional[Config] = None
        self._raw_config: Dict[str, Any] = {}
    
    def load_config(self) -> Config:
        """Load configuration from file and environment"""
        if self._config is None:
            self._load_raw_config()
            self._config = self._create_config_from_raw()
            self._validate_config()
        return self._config
    
    def _load_raw_config(self):
        """Load raw configuration from file"""
        # Default configuration
        self._raw_config = {
            "logging": {"level": "INFO", "console": True},
            "api": {"timeout": 30, "max_retries": 3},
            "gui": {"click_delay": 0.1, "typing_delay": 0.05},
            "security": {"coordinate_validation": True},
            "performance": {"max_concurrent_tasks": 1},
            "verification": {"enabled": True, "confidence_threshold": 0.8},
        }
        
        # Load from file if exists
        if self.config_path and self.config_path.exists():
            try:
                if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    with open(self.config_path, 'r') as f:
                        file_config = yaml.safe_load(f)
                elif self.config_path.suffix.lower() == '.json':
                    with open(self.config_path, 'r') as f:
                        file_config = json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported config file format: {self.config_path.suffix}",
                        config_file=str(self.config_path)
                    )
                
                # Merge with default config
                self._merge_config(self._raw_config, file_config)
                
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to load config file: {e}",
                    config_file=str(self.config_path)
                )
        
        # Override with environment variables
        self._load_from_environment()
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            "AI_AGENT_LOG_LEVEL": ("logging", "level"),
            "AI_AGENT_LOG_FILE": ("logging", "file"),
            "AI_AGENT_LOG_JSON": ("logging", "json_format"),
            "AI_AGENT_LOCAL_ENDPOINT": ("api", "local_endpoint"),
            "AI_AGENT_LOCAL_MODEL": ("api", "local_model"),
            "AI_AGENT_PREFERRED_PROVIDER": ("api", "preferred_provider"),
            "AI_AGENT_API_TIMEOUT": ("api", "timeout"),
            "AI_AGENT_API_MAX_RETRIES": ("api", "max_retries"),
            "AI_AGENT_CLICK_DELAY": ("gui", "click_delay"),
            "AI_AGENT_TYPING_DELAY": ("gui", "typing_delay"),
            "AI_AGENT_SCREENSHOT_QUALITY": ("gui", "screenshot_quality"),
            "AI_AGENT_MAX_CONCURRENT_TASKS": ("performance", "max_concurrent_tasks"),
            "AI_AGENT_TASK_TIMEOUT": ("performance", "task_timeout"),
            "AI_AGENT_VERIFICATION_ENABLED": ("verification", "enabled"),
            "AI_AGENT_VERIFICATION_CONFIDENCE_THRESHOLD": ("verification", "confidence_threshold"),
            "AI_AGENT_VERIFICATION_MAX_ATTEMPTS": ("verification", "max_verification_attempts"),
            "AI_AGENT_VERIFICATION_MAX_REGENERATIONS": ("verification", "max_regenerations"),
            "AI_AGENT_VERIFICATION_MODEL": ("verification", "verification_model"),
            "AI_AGENT_VERIFICATION_TIMEOUT": ("verification", "verification_timeout"),
            "AI_AGENT_VERIFICATION_AUTO_REGENERATE": ("verification", "auto_regenerate"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Type conversion
                if key in ["timeout", "max_retries", "click_delay", "typing_delay", 
                          "screenshot_quality", "max_concurrent_tasks", "task_timeout",
                          "max_verification_attempts", "max_regenerations", "verification_timeout"]:
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        continue
                elif key in ["json_format", "console", "coordinate_validation", "enabled", "auto_regenerate"]:
                    value = value.lower() in ['true', '1', 'yes', 'on']
                
                # Set in config
                if section not in self._raw_config:
                    self._raw_config[section] = {}
                self._raw_config[section][key] = value
    
    def _create_config_from_raw(self) -> Config:
        """Create Config object from raw configuration"""
        try:
            return Config(
                logging=LoggingConfig(**self._raw_config.get("logging", {})),
                api=APIConfig(**self._raw_config.get("api", {})),
                gui=GUIConfig(**self._raw_config.get("gui", {})),
                security=SecurityConfig(**self._raw_config.get("security", {})),
                performance=PerformanceConfig(**self._raw_config.get("performance", {})),
                verification=VerificationConfig(**self._raw_config.get("verification", {})),
                platform=self._raw_config.get("platform", {}),
                custom=self._raw_config.get("custom", {}),
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create config object: {e}",
                config_key="config_creation"
            )
    
    def _validate_config(self):
        """Validate configuration"""
        # Basic validation - no complex schema validation needed
        if not isinstance(self._raw_config, dict):
            raise ConfigurationError("Configuration must be a dictionary")
    
    def save_config(self, config_path: Optional[Union[str, Path]] = None):
        """Save current configuration to file"""
        if not self._config:
            self.load_config()
        
        save_path = Path(config_path) if config_path else self.config_path
        if not save_path:
            raise ConfigurationError("No config path specified for saving")
        
        # Convert config to dictionary
        config_dict = {
            "logging": {
                "level": self._config.logging.level,
                "file": self._config.logging.file,
                "json_format": self._config.logging.json_format,
                "console": self._config.logging.console,
                "max_file_size": self._config.logging.max_file_size,
                "backup_count": self._config.logging.backup_count,
            },
            "api": {
                "local_endpoint": self._config.api.local_endpoint,
                "local_model": self._config.api.local_model,
                # API key intentionally excluded to prevent saving sensitive data
                "preferred_provider": self._config.api.preferred_provider,
                "timeout": self._config.api.timeout,
                "max_retries": self._config.api.max_retries,
                "retry_delay": self._config.api.retry_delay,
            },
            "gui": {
                "click_delay": self._config.gui.click_delay,
                "typing_delay": self._config.gui.typing_delay,
                "scroll_duration": self._config.gui.scroll_duration,
                "drag_duration": self._config.gui.drag_duration,
                "screenshot_quality": self._config.gui.screenshot_quality,
                "screenshot_format": self._config.gui.screenshot_format,
                "coordinate_validation": self._config.gui.coordinate_validation,
            },
            "security": {
                "max_coordinate_value": self._config.security.max_coordinate_value,
                "min_coordinate_value": self._config.security.min_coordinate_value,
                "allowed_commands": self._config.security.allowed_commands,
                "sanitize_text_input": self._config.security.sanitize_text_input,
                "validate_file_paths": self._config.security.validate_file_paths,
                "max_text_length": self._config.security.max_text_length,
            },
            "performance": {
                "max_concurrent_tasks": self._config.performance.max_concurrent_tasks,
                "task_timeout": self._config.performance.task_timeout,
                "screenshot_timeout": self._config.performance.screenshot_timeout,
                "api_timeout": self._config.performance.api_timeout,
                "memory_limit_mb": self._config.performance.memory_limit_mb,
            },
            "platform": self._config.platform,
            "custom": self._config.custom,
        }
        
        # Save to file
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if save_path.suffix.lower() in ['.yaml', '.yml']:
                with open(save_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif save_path.suffix.lower() == '.json':
                with open(save_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                raise ConfigurationError(
                    f"Unsupported config file format: {save_path.suffix}",
                    config_file=str(save_path)
                )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save config file: {e}",
                config_file=str(save_path)
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        if not self._config:
            self.load_config()
        
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except (AttributeError, KeyError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value by dot notation key"""
        if not self._config:
            self.load_config()
        
        keys = key.split('.')
        config_obj = self._config
        
        # Navigate to parent
        for k in keys[:-1]:
            if hasattr(config_obj, k):
                config_obj = getattr(config_obj, k)
            elif isinstance(config_obj, dict):
                if k not in config_obj:
                    config_obj[k] = {}
                config_obj = config_obj[k]
        
        # Set value
        final_key = keys[-1]
        if hasattr(config_obj, final_key):
            setattr(config_obj, final_key, value)
        elif isinstance(config_obj, dict):
            config_obj[final_key] = value


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration (singleton pattern)"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    
    return _config_manager.load_config()


def get_config_manager() -> ConfigManager:
    """Get global config manager instance"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return _config_manager


def save_config(config_path: Optional[Union[str, Path]] = None):
    """Save current configuration"""
    config_manager = get_config_manager()
    config_manager.save_config(config_path)
