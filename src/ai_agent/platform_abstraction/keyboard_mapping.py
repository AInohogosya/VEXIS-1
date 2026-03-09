"""
OS-specific keyboard mapping and key detection
Zero-defect policy: comprehensive keyboard support with fallbacks
"""

import platform
import subprocess
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
from ..utils.exceptions import PlatformError
from ..utils.logger import get_logger


class KeyCategory(Enum):
    """Key categories for organization"""
    MODIFIER = "modifier"
    FUNCTION = "function"
    NAVIGATION = "navigation"
    MEDIA = "media"
    SYSTEM = "system"
    SPECIAL = "special"


@dataclass
class KeyMapping:
    """Key mapping definition"""
    name: str
    pyautogui_key: str
    category: KeyCategory
    description: str
    os_specific: bool = False
    alternatives: Optional[List[str]] = None


class KeyboardMapping:
    """OS-specific keyboard mapping detection and management"""
    
    def __init__(self):
        self.logger = get_logger("keyboard_mapping")
        self.system_info = self._detect_system_info()
        self._key_mappings = self._initialize_key_mappings()
        
    def _detect_system_info(self) -> Dict[str, str]:
        """Detect basic system information"""
        return {
            "os": platform.system().lower(),
            "platform": platform.platform().lower(),
        }
    
    def _initialize_key_mappings(self) -> Dict[str, KeyMapping]:
        """Initialize OS-specific key mappings"""
        mappings = {}
        
        # Common keys across all platforms
        common_keys = {
            # Basic modifiers
            "ctrl": KeyMapping("Ctrl", "ctrl", KeyCategory.MODIFIER, "Control key"),
            "alt": KeyMapping("Alt", "alt", KeyCategory.MODIFIER, "Alt key"),
            "shift": KeyMapping("Shift", "shift", KeyCategory.MODIFIER, "Shift key"),
            "space": KeyMapping("Space", "space", KeyCategory.NAVIGATION, "Space bar"),
            "enter": KeyMapping("Enter", "enter", KeyCategory.NAVIGATION, "Enter key"),
            "tab": KeyMapping("Tab", "tab", KeyCategory.NAVIGATION, "Tab key"),
            "esc": KeyMapping("Escape", "esc", KeyCategory.SYSTEM, "Escape key"),
            "backspace": KeyMapping("Backspace", "backspace", KeyCategory.NAVIGATION, "Backspace key"),
            "delete": KeyMapping("Delete", "delete", KeyCategory.NAVIGATION, "Delete key"),
            
            # Function keys
            "f1": KeyMapping("F1", "f1", KeyCategory.FUNCTION, "Function key 1"),
            "f2": KeyMapping("F2", "f2", KeyCategory.FUNCTION, "Function key 2"),
            "f3": KeyMapping("F3", "f3", KeyCategory.FUNCTION, "Function key 3"),
            "f4": KeyMapping("F4", "f4", KeyCategory.FUNCTION, "Function key 4"),
            "f5": KeyMapping("F5", "f5", KeyCategory.FUNCTION, "Function key 5"),
            "f6": KeyMapping("F6", "f6", KeyCategory.FUNCTION, "Function key 6"),
            "f7": KeyMapping("F7", "f7", KeyCategory.FUNCTION, "Function key 7"),
            "f8": KeyMapping("F8", "f8", KeyCategory.FUNCTION, "Function key 8"),
            "f9": KeyMapping("F9", "f9", KeyCategory.FUNCTION, "Function key 9"),
            "f10": KeyMapping("F10", "f10", KeyCategory.FUNCTION, "Function key 10"),
            "f11": KeyMapping("F11", "f11", KeyCategory.FUNCTION, "Function key 11"),
            "f12": KeyMapping("F12", "f12", KeyCategory.FUNCTION, "Function key 12"),
            
            # Navigation
            "up": KeyMapping("Up Arrow", "up", KeyCategory.NAVIGATION, "Up arrow key"),
            "down": KeyMapping("Down Arrow", "down", KeyCategory.NAVIGATION, "Down arrow key"),
            "left": KeyMapping("Left Arrow", "left", KeyCategory.NAVIGATION, "Left arrow key"),
            "right": KeyMapping("Right Arrow", "right", KeyCategory.NAVIGATION, "Right arrow key"),
            "home": KeyMapping("Home", "home", KeyCategory.NAVIGATION, "Home key"),
            "end": KeyMapping("End", "end", KeyCategory.NAVIGATION, "End key"),
            "pageup": KeyMapping("Page Up", "pageup", KeyCategory.NAVIGATION, "Page up key"),
            "pagedown": KeyMapping("Page Down", "pagedown", KeyCategory.NAVIGATION, "Page down key"),
            
            # Media keys
            "playpause": KeyMapping("Play/Pause", "playpause", KeyCategory.MEDIA, "Play/Pause media key"),
            "nexttrack": KeyMapping("Next Track", "nexttrack", KeyCategory.MEDIA, "Next track media key"),
            "prevtrack": KeyMapping("Previous Track", "prevtrack", KeyCategory.MEDIA, "Previous track media key"),
            "volumemute": KeyMapping("Volume Mute", "volumemute", KeyCategory.MEDIA, "Volume mute key"),
            "volumeup": KeyMapping("Volume Up", "volumeup", KeyCategory.MEDIA, "Volume up key"),
            "volumedown": KeyMapping("Volume Down", "volumedown", KeyCategory.MEDIA, "Volume down key"),
        }
        
        mappings.update(common_keys)
        
        # OS-specific keys
        if self.system_info["os"] == "darwin":  # macOS
            mac_keys = {
                "cmd": KeyMapping("Command", "command", KeyCategory.MODIFIER, "macOS Command key", True, ["meta"]),
                "option": KeyMapping("Option", "option", KeyCategory.MODIFIER, "macOS Option key", True, ["alt"]),
                "control": KeyMapping("Control", "control", KeyCategory.MODIFIER, "macOS Control key", True),
                "capslock": KeyMapping("Caps Lock", "capslock", KeyCategory.MODIFIER, "Caps Lock key", True),
                "fn": KeyMapping("Function", "fn", KeyCategory.MODIFIER, "Function key on Mac keyboards", True),
                "eject": KeyMapping("Eject", "eject", KeyCategory.SYSTEM, "Eject key on Mac keyboards", True),
                "power": KeyMapping("Power", "power", KeyCategory.SYSTEM, "Power key on Mac keyboards", True),
                "mission_control": KeyMapping("Mission Control", "f3", KeyCategory.SYSTEM, "Mission Control shortcut", True),
                "launchpad": KeyMapping("Launchpad", "f4", KeyCategory.SYSTEM, "Launchpad shortcut", True),
                "dashboard": KeyMapping("Dashboard", "f12", KeyCategory.SYSTEM, "Dashboard shortcut", True),
            }
            mappings.update(mac_keys)
            
        elif self.system_info["os"] == "windows":
            windows_keys = {
                "win": KeyMapping("Windows", "win", KeyCategory.MODIFIER, "Windows key", True, ["meta"]),
                "ctrl": KeyMapping("Control", "ctrl", KeyCategory.MODIFIER, "Control key", True),
                "alt": KeyMapping("Alt", "alt", KeyCategory.MODIFIER, "Alt key", True),
                "altgr": KeyMapping("Alt Gr", "altgr", KeyCategory.MODIFIER, "Alt Gr key", True),
                "menu": KeyMapping("Menu", "menu", KeyCategory.SYSTEM, "Menu key", True),
                "sleep": KeyMapping("Sleep", "sleep", KeyCategory.SYSTEM, "Sleep key", True),
                "wakeup": KeyMapping("Wake Up", "wakeup", KeyCategory.SYSTEM, "Wake up key", True),
                "copilot": KeyMapping("Copilot", "copilot", KeyCategory.SPECIAL, "Windows Copilot key", True),
                "printscreen": KeyMapping("Print Screen", "printscreen", KeyCategory.SYSTEM, "Print Screen key", True),
                "scrolllock": KeyMapping("Scroll Lock", "scrolllock", KeyCategory.SYSTEM, "Scroll Lock key", True),
                "pause": KeyMapping("Pause/Break", "pause", KeyCategory.SYSTEM, "Pause/Break key", True),
                "numlock": KeyMapping("Num Lock", "numlock", KeyCategory.MODIFIER, "Num Lock key", True),
                "apps": KeyMapping("Applications", "apps", KeyCategory.SYSTEM, "Applications key", True),
            }
            mappings.update(windows_keys)
            
        elif self.system_info["os"] == "linux":
            linux_keys = {
                "super": KeyMapping("Super", "super", KeyCategory.MODIFIER, "Linux Super key", True, ["meta", "win"]),
                "hyper": KeyMapping("Hyper", "hyper", KeyCategory.MODIFIER, "Hyper key", True),
                "ctrl": KeyMapping("Control", "ctrl", KeyCategory.MODIFIER, "Control key", True),
                "alt": KeyMapping("Alt", "alt", KeyCategory.MODIFIER, "Alt key", True),
                "altgr": KeyMapping("Alt Gr", "altgr", KeyCategory.MODIFIER, "Alt Gr key", True),
                "compose": KeyMapping("Compose", "compose", KeyCategory.MODIFIER, "Compose key", True),
                "printscreen": KeyMapping("Print Screen", "printscreen", KeyCategory.SYSTEM, "Print Screen key", True),
                "scrolllock": KeyMapping("Scroll Lock", "scrolllock", KeyCategory.SYSTEM, "Scroll Lock key", True),
                "pause": KeyMapping("Pause", "pause", KeyCategory.SYSTEM, "Pause key", True),
                "insert": KeyMapping("Insert", "insert", KeyCategory.NAVIGATION, "Insert key", True),
            }
            mappings.update(linux_keys)
        
        return mappings
    
    def get_key_mapping(self, key_name: str) -> Optional[KeyMapping]:
        """Get key mapping by name"""
        # Try exact match first
        if key_name.lower() in self._key_mappings:
            return self._key_mappings[key_name.lower()]
        
        # Try case-insensitive search
        for name, mapping in self._key_mappings.items():
            if name.lower() == key_name.lower():
                return mapping
        
        # Try alternative names
        for mapping in self._key_mappings.values():
            if mapping.alternatives:
                if key_name.lower() in [alt.lower() for alt in mapping.alternatives]:
                    return mapping
        
        return None
    
    def get_pyautogui_key(self, key_name: str) -> Optional[str]:
        """Get PyAutoGUI key name"""
        mapping = self.get_key_mapping(key_name)
        return mapping.pyautogui_key if mapping else None
    
    def get_keys_by_category(self, category: KeyCategory) -> List[KeyMapping]:
        """Get all keys in a category"""
        return [mapping for mapping in self._key_mappings.values() if mapping.category == category]
    
    def get_os_specific_keys(self) -> List[KeyMapping]:
        """Get OS-specific keys"""
        return [mapping for mapping in self._key_mappings.values() if mapping.os_specific]
    
    def get_common_keys(self) -> List[KeyMapping]:
        """Get common keys across all platforms"""
        return [mapping for mapping in self._key_mappings.values() if not mapping.os_specific]
    
    def normalize_key_combination(self, keys: str) -> str:
        """Normalize key combination to PyAutoGUI format"""
        if not keys:
            return ""
        
        key_list = [key.strip() for key in keys.split('+')]
        normalized_keys = []
        
        for key in key_list:
            mapping = self.get_key_mapping(key)
            if mapping:
                normalized_keys.append(mapping.pyautogui_key)
            else:
                # Try to use the key as-is
                normalized_keys.append(key.lower())
        
        return '+'.join(normalized_keys)
    
    def detect_available_keys(self) -> Set[str]:
        """Detect available keys on the current system"""
        available_keys = set()
        
        try:
            if self.system_info["os"] == "darwin":
                available_keys.update(self._detect_macos_keys())
            elif self.system_info["os"] == "windows":
                available_keys.update(self._detect_windows_keys())
            elif self.system_info["os"] == "linux":
                available_keys.update(self._detect_linux_keys())
        except Exception as e:
            self.logger.warning(f"Key detection failed: {e}")
        
        return available_keys
    
    def _detect_macos_keys(self) -> Set[str]:
        """Detect available keys on macOS"""
        available_keys = set()
        
        try:
            # Check for special Mac keys using system_profiler
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                # Check for Mac keyboard type
                hardware = data.get("SPHardwareDataType", [{}])[0]
                keyboard_type = hardware.get("keyboard_type", "")
                
                if "laptop" in keyboard_type.lower() or "notebook" in keyboard_type.lower():
                    available_keys.update(["fn", "power"])
                
                # Most Mac keyboards have these keys
                available_keys.update(["cmd", "option", "control", "capslock"])
                
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            # Fallback to common Mac keys
            available_keys.update(["cmd", "option", "control", "capslock", "fn"])
        
        return available_keys
    
    def _detect_windows_keys(self) -> Set[str]:
        """Detect available keys on Windows"""
        available_keys = set()
        
        try:
            # Check for Windows-specific keys
            result = subprocess.run(
                ["powershell", "-Command", "Get-WmiObject -Class Win32_Keyboard | Select-Object -Property Name"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                keyboard_info = result.stdout.lower()
                
                # Check for Copilot key support (newer keyboards)
                if "copilot" in keyboard_info:
                    available_keys.add("copilot")
                
                # Most Windows keyboards have these keys
                available_keys.update(["win", "ctrl", "alt", "menu", "printscreen"])
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback to common Windows keys
            available_keys.update(["win", "ctrl", "alt", "menu", "printscreen"])
        
        return available_keys
    
    def _detect_linux_keys(self) -> Set[str]:
        """Detect available keys on Linux"""
        available_keys = set()
        
        try:
            # Check for keyboard layout and special keys
            result = subprocess.run(
                ["setxkbmap", "-query"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                layout_info = result.stdout.lower()
                
                # Most Linux keyboards have these keys
                available_keys.update(["super", "ctrl", "alt", "compose"])
                
                # Check for special keys
                if "compose" in layout_info:
                    available_keys.add("compose")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback to common Linux keys
            available_keys.update(["super", "ctrl", "alt", "compose"])
        
        return available_keys
    
    def get_keyboard_info(self) -> Dict[str, any]:
        """Get comprehensive keyboard information"""
        return {
            "os": self.system_info["os"],
            "platform": self.system_info["platform"],
            "total_keys": len(self._key_mappings),
            "os_specific_keys": len(self.get_os_specific_keys()),
            "common_keys": len(self.get_common_keys()),
            "available_keys": list(self.detect_available_keys()),
            "categories": {
                category.value: len(self.get_keys_by_category(category))
                for category in KeyCategory
            }
        }


# Global keyboard mapping instance
_keyboard_mapping: Optional[KeyboardMapping] = None


def get_keyboard_mapping() -> KeyboardMapping:
    """Get global keyboard mapping instance"""
    global _keyboard_mapping
    
    if _keyboard_mapping is None:
        _keyboard_mapping = KeyboardMapping()
    
    return _keyboard_mapping
