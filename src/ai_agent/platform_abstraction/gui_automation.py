"""
Cross-platform GUI automation with fallback mechanisms
Zero-defect policy: comprehensive automation with error recovery
"""

import time
import math
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum

from .platform_detector import get_system_info, get_platform_detector
from .keyboard_mapping import get_keyboard_mapping
from ..utils.exceptions import ExecutionError, PlatformError, ValidationError
from ..utils.logger import get_logger


class MouseAction(Enum):
    """Mouse action types"""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    SCROLL = "scroll"


class KeyAction(Enum):
    """Key action types"""
    PRESS = "press"
    RELEASE = "release"
    TYPE = "type"


@dataclass
class AutomationResult:
    """Result of automation operation"""
    success: bool
    action: str
    duration: float
    method: str
    error: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class GUIAutomation:
    """Cross-platform GUI automation with fallback mechanisms"""
    
    def __init__(
        self,
        click_delay: float = 0.1,
        typing_delay: float = 0.05,
        scroll_duration: float = 0.5,
        drag_duration: float = 0.3
    ):
        self.click_delay = click_delay
        self.typing_delay = typing_delay
        self.scroll_duration = scroll_duration
        self.drag_duration = drag_duration
        
        self.logger = get_logger("gui_automation")
        
        # Get system information
        self.system_info = get_system_info()
        self.platform_detector = get_platform_detector()
        
        # Initialize keyboard mapping
        self.keyboard_mapping = get_keyboard_mapping()
        
        # Initialize automation methods in order of preference
        self._automation_methods = self._initialize_automation_methods()
        
        # Current method index (for fallback)
        self._current_method_index = 0
        
        self.logger.info(
            "GUI automation initialized",
            platform=self.system_info.os_name,
            click_delay=self.click_delay,
            typing_delay=self.typing_delay,
            keyboard_keys=len(self.keyboard_mapping._key_mappings),
            os_specific_keys=len(self.keyboard_mapping.get_os_specific_keys()),
            methods=[getattr(method, '__name__', method.__class__.__name__) for method in self._automation_methods],
        )
    
    def _initialize_automation_methods(self) -> List[callable]:
        """Initialize single automation method"""
        methods = []
        
        # Use only pyautogui for simplicity
        methods.append(self._automation_pyautogui())
        
        return methods
    
    def click(self, x: float, y: float) -> AutomationResult:
        """Perform click at normalized coordinates"""
        return self._execute_with_fallback(
            "click",
            lambda method: method.click(x, y),
            x=x, y=y
        )
    
    def double_click(self, x: float, y: float) -> AutomationResult:
        """Perform double-click at normalized coordinates"""
        return self._execute_with_fallback(
            "double_click",
            lambda method: method.double_click(x, y),
            x=x, y=y
        )
    
    def right_click(self, x: float, y: float) -> AutomationResult:
        """Perform right-click at normalized coordinates"""
        return self._execute_with_fallback(
            "right_click",
            lambda method: method.right_click(x, y),
            x=x, y=y
        )
    
    def drag(self, start_x: float, start_y: float, end_x: float, end_y: float) -> AutomationResult:
        """Perform drag from start to end coordinates"""
        return self._execute_with_fallback(
            "drag",
            lambda method: method.drag(start_x, start_y, end_x, end_y),
            start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y
        )
    
    def scroll(self, direction: str, amount: int) -> AutomationResult:
        """Perform scroll operation"""
        return self._execute_with_fallback(
            "scroll",
            lambda method: method.scroll(direction, amount),
            direction=direction, amount=amount
        )
    
    def type_text(self, text: str) -> AutomationResult:
        """Type text"""
        return self._execute_with_fallback(
            "type_text",
            lambda method: method.type_text(text),
            text=text
        )
    
    def press_keys(self, keys: str) -> AutomationResult:
        """Press key combination with OS-specific mapping"""
        # Normalize key combination using keyboard mapping
        normalized_keys = self.keyboard_mapping.normalize_key_combination(keys)
        
        self.logger.debug(
            "Key combination normalized",
            original=keys,
            normalized=normalized_keys,
            platform=self.system_info.os_name
        )
        
        return self._execute_with_fallback(
            "press_keys",
            lambda method: method.press_keys(normalized_keys),
            keys=normalized_keys
        )
    
    def _execute_with_fallback(self, action: str, operation: callable, **kwargs) -> AutomationResult:
        """Execute operation without fallback"""
        start_time = time.time()
        
        # Basic input validation
        if action in ["click", "double_click", "right_click"]:
            if not (0.0 <= kwargs.get('x', 0) <= 1.0) or not (0.0 <= kwargs.get('y', 0) <= 1.0):
                raise ValidationError("Coordinates must be between 0.0 and 1.0")
            
        elif action == "drag":
            if not (0.0 <= kwargs.get('start_x', 0) <= 1.0) or not (0.0 <= kwargs.get('start_y', 0) <= 1.0):
                raise ValidationError("Start coordinates must be between 0.0 and 1.0")
            if not (0.0 <= kwargs.get('end_x', 0) <= 1.0) or not (0.0 <= kwargs.get('end_y', 0) <= 1.0):
                raise ValidationError("End coordinates must be between 0.0 and 1.0")
            
        elif action == "scroll":
            if kwargs.get('direction') not in ["up", "down", "left", "right"]:
                raise ValidationError("Invalid scroll direction")
            if not (1 <= kwargs.get('amount', 1) <= 10):
                raise ValidationError("Scroll amount must be between 1 and 10")
        
        # Execute with single method
        method = self._automation_methods[0]
        
        try:
            result = operation(method)
            duration = time.time() - start_time
            
            return AutomationResult(
                success=True,
                action=action,
                duration=duration,
                method="pyautogui",
                coordinates=(kwargs.get('x'), kwargs.get('y')) or (kwargs.get('start_x'), kwargs.get('start_y')),
                metadata=kwargs,
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Automation failed for {action}: {e}"
            
            return AutomationResult(
                success=False,
                action=action,
                duration=duration,
                method="pyautogui",
                error=error_msg,
                metadata=kwargs,
            )
    
    def _get_absolute_coordinates(self, x: float, y: float) -> Tuple[int, int]:
        """Convert normalized coordinates to absolute coordinates"""
        if not self.system_info.screen_resolution:
            raise ExecutionError("Screen resolution not available")
        
        screen_width, screen_height = self.system_info.screen_resolution
        abs_x = int(x * screen_width)
        abs_y = int(y * screen_height)
        
        return abs_x, abs_y
    
    def get_available_keys(self) -> List[str]:
        """Get list of available keys for current platform"""
        return list(self.keyboard_mapping.detect_available_keys())
    
    def get_key_info(self, key_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific key"""
        mapping = self.keyboard_mapping.get_key_mapping(key_name)
        if mapping:
            return {
                "name": mapping.name,
                "pyautogui_key": mapping.pyautogui_key,
                "category": mapping.category.value,
                "description": mapping.description,
                "os_specific": mapping.os_specific,
                "alternatives": mapping.alternatives,
            }
        return None
    
    def is_key_available(self, key_name: str) -> bool:
        """Check if a key is available on the current platform"""
        available_keys = self.get_available_keys()
        normalized_key = self.keyboard_mapping.get_pyautogui_key(key_name)
        return normalized_key in available_keys if normalized_key else False
    
    # Platform-specific automation methods
    
    def _automation_pyautogui(self) -> 'PyAutoGUIAutomation':
        """PyAutoGUI automation (single method)"""
        return PyAutoGUIAutomation(self)


# Cross-platform automation implementations

class PyAutoGUIAutomation:
    """PyAutoGUI automation (cross-platform)"""
    
    def __init__(self, parent: GUIAutomation):
        self.parent = parent
        self.logger = get_logger("automation_pyautogui")
        
        # Configure PyAutoGUI
        import pyautogui
        pyautogui.PAUSE = 0.1
        pyautogui.FAILSAFE = True
    
    def click(self, x: float, y: float):
        import pyautogui
        abs_x, abs_y = self.parent._get_absolute_coordinates(x, y)
        pyautogui.click(abs_x, abs_y)
        time.sleep(self.parent.click_delay)
    
    def double_click(self, x: float, y: float):
        import pyautogui
        abs_x, abs_y = self.parent._get_absolute_coordinates(x, y)
        pyautogui.doubleClick(abs_x, abs_y)
        time.sleep(self.parent.click_delay)
    
    def right_click(self, x: float, y: float):
        import pyautogui
        abs_x, abs_y = self.parent._get_absolute_coordinates(x, y)
        pyautogui.rightClick(abs_x, abs_y)
        time.sleep(self.parent.click_delay)
    
    def drag(self, start_x: float, start_y: float, end_x: float, end_y: float):
        import pyautogui
        start_abs_x, start_abs_y = self.parent._get_absolute_coordinates(start_x, start_y)
        end_abs_x, end_abs_y = self.parent._get_absolute_coordinates(end_x, end_y)
        pyautogui.dragTo(end_abs_x, end_abs_y, duration=self.parent.drag_duration)
        time.sleep(0.1)
    
    def scroll(self, direction: str, amount: int):
        import pyautogui
        if direction == "up":
            pyautogui.scroll(amount)
        elif direction == "down":
            pyautogui.scroll(-amount)
        else:
            pyautogui.hscroll(amount if direction == "right" else -amount)
        time.sleep(0.1)
    
    def type_text(self, text: str):
        import pyautogui
        pyautogui.typewrite(text, interval=self.parent.typing_delay)
    
    def press_keys(self, keys: str):
        import pyautogui
        
        # Handle special OS-specific keys
        if '+' in keys:
            # Key combination
            key_list = keys.split('+')
            pyautogui.hotkey(*key_list)
        else:
            # Single key
            try:
                pyautogui.press(keys)
            except Exception:
                # Fallback for special keys that might not be recognized
                self.logger.warning(f"Key '{keys}' not recognized by PyAutoGUI")
                # Try to type the key as a character
                pyautogui.typewrite(keys, interval=0.05)
        
        time.sleep(0.1)
