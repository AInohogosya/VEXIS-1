"""
Platform detection and system information
Zero-defect policy: comprehensive platform detection with fallbacks
"""

import platform
import sys
import os
import subprocess
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ..utils.exceptions import PlatformError
from ..utils.logger import get_logger
from .keyboard_mapping import get_keyboard_mapping


@dataclass
class SystemInfo:
    """System information structure"""
    os_name: str
    os_version: str
    architecture: str
    platform: str
    python_version: str
    screen_resolution: Optional[Tuple[int, int]] = None
    scale_factor: float = 1.0
    display_count: int = 1
    is_headless: bool = False
    is_container: bool = False
    is_virtual_machine: bool = False
    keyboard_info: Optional[Dict[str, Any]] = None


class PlatformDetector:
    """Comprehensive platform detection with fallback mechanisms"""
    
    def __init__(self):
        self.logger = get_logger("platform_detector")
        self._system_info: Optional[SystemInfo] = None
    
    def detect_system(self) -> SystemInfo:
        """Detect complete system information"""
        if self._system_info is None:
            self._system_info = self._perform_detection()
        return self._system_info
    
    def _perform_detection(self) -> SystemInfo:
        """Perform comprehensive system detection"""
        self.logger.info("Starting platform detection...")
        
        try:
            # Basic platform detection
            os_name, os_version = self._detect_os()
            architecture = self._detect_architecture()
            platform_name = self._detect_platform()
            python_version = self._detect_python_version()
            
            # Display detection
            screen_resolution = self._detect_screen_resolution()
            scale_factor = self._detect_scale_factor()
            display_count = self._detect_display_count()
            
            # Environment detection
            is_headless = self._detect_headless()
            is_container = self._detect_container()
            is_virtual_machine = self._detect_virtual_machine()
            
            # Keyboard detection
            keyboard_info = self._detect_keyboard_info()
            
            system_info = SystemInfo(
                os_name=os_name,
                os_version=os_version,
                architecture=architecture,
                platform=platform_name,
                python_version=python_version,
                screen_resolution=screen_resolution,
                scale_factor=scale_factor,
                display_count=display_count,
                is_headless=is_headless,
                is_container=is_container,
                is_virtual_machine=is_virtual_machine,
                keyboard_info=keyboard_info,
            )
            
            self.logger.info(
                "Platform detection completed",
                os_name=os_name,
                os_version=os_version,
                architecture=architecture,
                platform=platform_name,
                screen_resolution=screen_resolution,
                scale_factor=scale_factor,
                display_count=display_count,
                is_headless=is_headless,
                is_container=is_container,
                is_virtual_machine=is_virtual_machine,
                keyboard_keys=len(keyboard_info.get("available_keys", [])) if keyboard_info else 0,
            )
            
            return system_info
            
        except Exception as e:
            self.logger.error(f"Platform detection failed: {e}")
            raise PlatformError(f"Failed to detect platform: {e}")
    
    def _detect_os(self) -> Tuple[str, str]:
        """Detect operating system and version"""
        try:
            # Primary method: platform module
            system = platform.system()
            release = platform.release()
            version = platform.version()
            
            if system == "Darwin":
                # macOS
                try:
                    result = subprocess.run(
                        ["sw_vers", "-productVersion"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        return "macos", result.stdout.strip()
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
                return "macos", release
                
            elif system == "Windows":
                # Windows
                try:
                    result = subprocess.run(
                        ["cmd", "/c", "ver"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        version_line = result.stdout.strip()
                        # Parse Windows version from ver output
                        if "Windows" in version_line:
                            version_part = version_line.split("Windows")[-1].strip()
                            return "windows", version_part
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
                return "windows", f"{release} {version}"
                
            elif system == "Linux":
                # Linux - try to get distribution info
                try:
                    with open("/etc/os-release", "r") as f:
                        content = f.read()
                        for line in content.split("\n"):
                            if line.startswith("ID="):
                                distro = line.split("=")[1].strip('"')
                            elif line.startswith("VERSION_ID="):
                                version_id = line.split("=")[1].strip('"')
                        return distro, version_id
                except (FileNotFoundError, KeyError):
                    pass
                return "linux", release
                
            else:
                # Fallback for other systems
                return system.lower(), release
                
        except Exception as e:
            self.logger.warning(f"OS detection failed: {e}")
            return "unknown", "unknown"
    
    def _detect_architecture(self) -> str:
        """Detect system architecture"""
        try:
            # Primary method: platform.machine()
            arch = platform.machine()
            if arch:
                return arch.lower()
            
            # Fallback: uname
            if hasattr(os, 'uname'):
                arch = os.uname().machine
                return arch.lower()
            
            # Fallback: sys.platform
            if sys.platform == "win32":
                return "x86_64" if sys.maxsize > 2**32 else "x86"
            
            return "unknown"
            
        except Exception as e:
            self.logger.warning(f"Architecture detection failed: {e}")
            return "unknown"
    
    def _detect_platform(self) -> str:
        """Detect platform type"""
        try:
            return sys.platform.lower()
        except Exception as e:
            self.logger.warning(f"Platform detection failed: {e}")
            return "unknown"
    
    def _detect_python_version(self) -> str:
        """Detect Python version"""
        try:
            return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        except Exception as e:
            self.logger.warning(f"Python version detection failed: {e}")
            return "unknown"
    
    def _detect_screen_resolution(self) -> Optional[Tuple[int, int]]:
        """Detect screen resolution"""
        try:
            # Try different methods based on platform
            if sys.platform == "darwin":  # macOS
                return self._detect_screen_resolution_macos()
            elif sys.platform == "win32":  # Windows
                return self._detect_screen_resolution_windows()
            elif sys.platform.startswith("linux"):  # Linux
                return self._detect_screen_resolution_linux()
            else:
                # Fallback using pyautogui
                import pyautogui
                width, height = pyautogui.size()
                return (width, height)
                
        except Exception as e:
            self.logger.warning(f"Screen resolution detection failed: {e}")
            return None
    
    def _detect_screen_resolution_macos(self) -> Optional[Tuple[int, int]]:
        """Detect screen resolution on macOS"""
        try:
            # Use system_profiler
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                for display in data.get("SPDisplaysDataType", []):
                    resolution = display.get("spdisplays_resolution")
                    if resolution:
                        # Parse resolution like "1920 x 1080"
                        parts = resolution.split(" x ")
                        if len(parts) == 2:
                            width = int(parts[0])
                            height = int(parts[1])
                            return (width, height)
            
            # Fallback: use Quartz/CoreGraphics
            try:
                from Quartz import CGDisplayBounds
                from Quartz import CGMainDisplayID
                
                display_id = CGMainDisplayID()
                bounds = CGDisplayBounds(display_id)
                width = int(bounds.size.width)
                height = int(bounds.size.height)
                return (width, height)
                
            except ImportError:
                pass
            
            return None
            
        except Exception as e:
            self.logger.warning(f"macOS screen resolution detection failed: {e}")
            return None
    
    def _detect_screen_resolution_windows(self) -> Optional[Tuple[int, int]]:
        """Detect screen resolution on Windows"""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            
            # Get screen dimensions
            width = user32.GetSystemMetrics(0)  # SM_CXSCREEN
            height = user32.GetSystemMetrics(1)  # SM_CYSCREEN
            
            return (width, height)
            
        except Exception as e:
            self.logger.warning(f"Windows screen resolution detection failed: {e}")
            return None
    
    def _detect_screen_resolution_linux(self) -> Optional[Tuple[int, int]]:
        """Detect screen resolution on Linux"""
        try:
            # Try xrandr
            result = subprocess.run(
                ["xrandr"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if " connected" in line and "*" in line:
                        # Parse resolution like "1920x1080"
                        parts = line.split()[0].split("x")
                        if len(parts) == 2:
                            width = int(parts[0])
                            height = int(parts[1])
                            return (width, height)
            
            # Fallback: try reading from /sys/class/drm
            drm_path = "/sys/class/drm"
            if os.path.exists(drm_path):
                for item in os.listdir(drm_path):
                    modes_path = os.path.join(drm_path, item, "modes")
                    if os.path.exists(modes_path):
                        with open(modes_path, "r") as f:
                            modes = f.read().strip()
                            if modes:
                                first_mode = modes.split("\n")[0]
                                if "x" in first_mode:
                                    width, height = map(int, first_mode.split("x"))
                                    return (width, height)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Linux screen resolution detection failed: {e}")
            return None
    
    def _detect_scale_factor(self) -> float:
        """Detect display scale factor"""
        try:
            if sys.platform == "darwin":  # macOS
                return self._detect_scale_factor_macos()
            elif sys.platform == "win32":  # Windows
                return self._detect_scale_factor_windows()
            elif sys.platform.startswith("linux"):  # Linux
                return self._detect_scale_factor_linux()
            else:
                return 1.0
                
        except Exception as e:
            self.logger.warning(f"Scale factor detection failed: {e}")
            return 1.0
    
    def _detect_scale_factor_macos(self) -> float:
        """Detect scale factor on macOS"""
        try:
            from Quartz import CGDisplayBounds
            from Quartz import CGMainDisplayID
            from Quartz import CGDisplayScreenSize
            from Quartz import CGDisplayPixelsHigh
            
            display_id = CGMainDisplayID()
            bounds = CGDisplayBounds(display_id)
            pixel_height = CGDisplayPixelsHigh(display_id)
            physical_height = CGDisplayScreenSize(display_id).height
            
            # Calculate DPI and scale factor
            if physical_height > 0:
                dpi = pixel_height / (physical_height / 25.4)  # Convert mm to inches
                scale_factor = dpi / 96.0  # Standard DPI
                return max(1.0, round(scale_factor, 1))
            
            return 1.0
            
        except Exception as e:
            self.logger.warning(f"macOS scale factor detection failed: {e}")
            return 1.0
    
    def _detect_scale_factor_windows(self) -> float:
        """Detect scale factor on Windows"""
        try:
            import ctypes
            
            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32
            
            # Get device context
            hdc = user32.GetDC(0)
            
            # Get logical and physical pixels
            logical_pixels_x = gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
            physical_pixels_x = gdi32.GetDeviceCaps(hdc, 118)  # DESKTOPHORZRES
            
            # Calculate scale factor
            if logical_pixels_x > 0:
                scale_factor = physical_pixels_x / logical_pixels_x
                return max(1.0, round(scale_factor, 1))
            
            return 1.0
            
        except Exception as e:
            self.logger.warning(f"Windows scale factor detection failed: {e}")
            return 1.0
    
    def _detect_scale_factor_linux(self) -> float:
        """Detect scale factor on Linux"""
        try:
            # Try reading from GTK settings
            try:
                result = subprocess.run(
                    ["gsettings", "get", "org.gnome.desktop.interface", "text-scaling-factor"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    scale_factor = float(result.stdout.strip())
                    return max(1.0, scale_factor)
                    
            except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
                pass
            
            # Try reading from Xft.dpi
            try:
                result = subprocess.run(
                    ["xrdb", "-query"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if line.startswith("Xft.dpi:"):
                            dpi = float(line.split(":")[1].strip())
                            scale_factor = dpi / 96.0
                            return max(1.0, scale_factor)
                            
            except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
                pass
            
            return 1.0
            
        except Exception as e:
            self.logger.warning(f"Linux scale factor detection failed: {e}")
            return 1.0
    
    def _detect_display_count(self) -> int:
        """Detect number of displays"""
        try:
            if sys.platform == "darwin":  # macOS
                return self._detect_display_count_macos()
            elif sys.platform == "win32":  # Windows
                return self._detect_display_count_windows()
            elif sys.platform.startswith("linux"):  # Linux
                return self._detect_display_count_linux()
            else:
                return 1
                
        except Exception as e:
            self.logger.warning(f"Display count detection failed: {e}")
            return 1
    
    def _detect_display_count_macos(self) -> int:
        """Detect display count on macOS"""
        try:
            from Quartz import CGGetActiveDisplayList
            
            result = CGGetActiveDisplayList(32, None, None)  # Max 32 displays
            return result[1]  # Return count
            
        except Exception as e:
            self.logger.warning(f"macOS display count detection failed: {e}")
            return 1
    
    def _detect_display_count_windows(self) -> int:
        """Detect display count on Windows"""
        try:
            import ctypes
            
            user32 = ctypes.windll.user32
            
            # Get display count
            count = user32.GetSystemMetrics(80)  # SM_CMONITORS
            return max(1, count)
            
        except Exception as e:
            self.logger.warning(f"Windows display count detection failed: {e}")
            return 1
    
    def _detect_display_count_linux(self) -> int:
        """Detect display count on Linux"""
        try:
            # Try xrandr
            result = subprocess.run(
                ["xrandr"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                connected_displays = 0
                for line in result.stdout.split("\n"):
                    if " connected" in line:
                        connected_displays += 1
                return max(1, connected_displays)
            
            return 1
            
        except Exception as e:
            self.logger.warning(f"Linux display count detection failed: {e}")
            return 1
    
    def _detect_headless(self) -> bool:
        """Detect if running in headless mode"""
        try:
            # Check display environment variables
            if not os.getenv("DISPLAY") and sys.platform.startswith("linux"):
                return True
            
            # Check for headless indicators
            headless_indicators = [
                "HEADLESS",
                "NO_DISPLAY",
                "CI",
                "GITHUB_ACTIONS",
                "JENKINS_URL",
                "TRAVIS",
            ]
            
            for indicator in headless_indicators:
                if os.getenv(indicator):
                    return True
            
            # Try to connect to display
            if sys.platform.startswith("linux"):
                try:
                    result = subprocess.run(
                        ["xset", "q"],
                        capture_output=True,
                        timeout=2
                    )
                    return result.returncode != 0
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Headless detection failed: {e}")
            return False
    
    def _detect_container(self) -> bool:
        """Detect if running in a container"""
        try:
            # Check for container indicators
            container_indicators = [
                "/.dockerenv",
                "/.dockerinit",
                "/proc/1/cgroup",
                "/proc/self/cgroup",
            ]
            
            for indicator in container_indicators:
                if os.path.exists(indicator):
                    return True
            
            # Check environment variables
            container_env_vars = [
                "DOCKER_CONTAINER",
                "KUBERNETES_SERVICE_HOST",
                "CONTAINER",
            ]
            
            for env_var in container_env_vars:
                if os.getenv(env_var):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Container detection failed: {e}")
            return False
    
    def _detect_virtual_machine(self) -> bool:
        """Detect if running in a virtual machine"""
        try:
            # Check for common VM indicators
            vm_indicators = [
                "VMware",
                "VirtualBox",
                "QEMU",
                "KVM",
                "Xen",
                "Hyper-V",
                "Parallels",
            ]
            
            # Check system info
            if sys.platform == "win32":
                try:
                    result = subprocess.run(
                        ["systeminfo"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0:
                        output = result.stdout.upper()
                        for indicator in vm_indicators:
                            if indicator.upper() in output:
                                return True
                                
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
            
            elif sys.platform.startswith("linux"):
                try:
                    # Check dmesg
                    result = subprocess.run(
                        ["dmesg"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.returncode == 0:
                        output = result.stdout.upper()
                        for indicator in vm_indicators:
                            if indicator.upper() in output:
                                return True
                                
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
                
                # Check CPU info
                try:
                    with open("/proc/cpuinfo", "r") as f:
                        cpuinfo = f.read().upper()
                        for indicator in vm_indicators:
                            if indicator.upper() in cpuinfo:
                                return True
                except FileNotFoundError:
                    pass
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Virtual machine detection failed: {e}")
            return False
    
    def _detect_keyboard_info(self) -> Dict[str, Any]:
        """Detect keyboard information and available keys"""
        try:
            keyboard_mapping = get_keyboard_mapping()
            return keyboard_mapping.get_keyboard_info()
        except Exception as e:
            self.logger.warning(f"Keyboard detection failed: {e}")
            return {
                "os": platform.system().lower(),
                "total_keys": 0,
                "os_specific_keys": 0,
                "common_keys": 0,
                "available_keys": [],
                "categories": {},
                "error": str(e)
            }
    
    def get_platform_specific_config(self) -> Dict[str, Any]:
        """Get platform-specific configuration"""
        system_info = self.detect_system()
        
        config = {
            "os_name": system_info.os_name,
            "platform": system_info.platform,
            "architecture": system_info.architecture,
        }
        
        # Add platform-specific settings
        if system_info.os_name == "macos":
            config.update({
                "screenshot_method": "quartz",
                "automation_method": "applescript",
                "font_family": "SF Pro Display",
                "shell": "/bin/bash",
            })
        elif system_info.os_name == "windows":
            config.update({
                "screenshot_method": "win32",
                "automation_method": "win32",
                "font_family": "Segoe UI",
                "shell": "cmd.exe",
            })
        elif system_info.os_name == "linux":
            config.update({
                "screenshot_method": "x11",
                "automation_method": "x11",
                "font_family": "DejaVu Sans",
                "shell": "/bin/bash",
            })
        
        # Add display settings
        if system_info.screen_resolution:
            config.update({
                "screen_width": system_info.screen_resolution[0],
                "screen_height": system_info.screen_resolution[1],
                "scale_factor": system_info.scale_factor,
                "display_count": system_info.display_count,
            })
        
        # Add environment settings
        config.update({
            "is_headless": system_info.is_headless,
            "is_container": system_info.is_container,
            "is_virtual_machine": system_info.is_virtual_machine,
        })
        
        # Add keyboard configuration
        if system_info.keyboard_info:
            config.update({
                "keyboard_info": system_info.keyboard_info,
                "os_specific_keys": system_info.keyboard_info.get("available_keys", []),
                "keyboard_categories": system_info.keyboard_info.get("categories", {}),
            })
        
        return config


# Global platform detector instance
_platform_detector: Optional[PlatformDetector] = None


def get_platform_detector() -> PlatformDetector:
    """Get global platform detector instance"""
    global _platform_detector
    
    if _platform_detector is None:
        _platform_detector = PlatformDetector()
    
    return _platform_detector


def get_system_info() -> SystemInfo:
    """Get system information"""
    detector = get_platform_detector()
    return detector.detect_system()
