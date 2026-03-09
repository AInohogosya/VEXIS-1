"""
Cross-platform screenshot capture with enhanced metadata
Zero-defect policy: comprehensive capture with fallback mechanisms
"""

import os
import io
import time
import tempfile
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from .platform_detector import get_system_info, get_platform_detector
from ..utils.exceptions import ScreenshotError, PlatformError
from ..utils.logger import get_logger


@dataclass
class ScreenshotMetadata:
    """Screenshot metadata structure"""
    timestamp: datetime
    width: int
    height: int
    format: str
    quality: int
    file_size: int
    capture_method: str
    platform: str
    scale_factor: float
    display_index: int
    color_depth: int
    dpi: Optional[Tuple[int, int]] = None
    compression_ratio: Optional[float] = None


class ScreenshotCapture:
    """Cross-platform screenshot capture with fallback mechanisms"""
    
    def __init__(self, quality: int = 95, format: str = "PNG"):
        self.quality = quality
        self.format = format.upper()
        self.logger = get_logger("screenshot_capture")
        
        # Validate format
        if self.format not in ["PNG", "JPEG", "WEBP"]:
            raise ScreenshotError(f"Unsupported format: {self.format}")
        
        # Get system information
        self.system_info = get_system_info()
        self.platform_detector = get_platform_detector()
        
        # Initialize capture methods in order of preference
        self._capture_methods = self._initialize_capture_methods()
        
        self.logger.info(
            "Screenshot capture initialized",
            format=self.format,
            quality=self.quality,
            platform=self.system_info.os_name,
            methods=[method.__name__ for method in self._capture_methods],
        )
    
    def _initialize_capture_methods(self) -> List[callable]:
        """Initialize platform-specific capture methods"""
        methods = []
        
        if self.system_info.os_name == "macos":
            methods.extend([
                self._capture_quartz,
                self._capture_pil,
                self._capture_opencv,
                self._capture_pyautogui,
            ])
        elif self.system_info.os_name == "windows":
            methods.extend([
                self._capture_win32,
                self._capture_pil,
                self._capture_opencv,
                self._capture_pyautogui,
            ])
        elif self.system_info.os_name == "linux":
            methods.extend([
                self._capture_x11,
                self._capture_wayland,
                self._capture_pil,
                self._capture_opencv,
                self._capture_pyautogui,
            ])
        else:
            # Generic fallback methods
            methods.extend([
                self._capture_pil,
                self._capture_opencv,
                self._capture_pyautogui,
            ])
        
        return methods
    
    def capture_screenshot(
        self,
        save_path: Optional[str] = None,
        display_index: int = 0,
        include_metadata: bool = True
    ) -> Tuple[bytes, ScreenshotMetadata]:
        """Capture screenshot with automatic fallback"""
        self.logger.info(
            "Starting screenshot capture",
            display_index=display_index,
            save_path=save_path,
        )
        
        last_error = None
        
        # Try each capture method
        for i, method in enumerate(self._capture_methods):
            try:
                self.logger.debug(
                    f"Attempting capture method {i+1}/{len(self._capture_methods)}",
                    method=method.__name__
                )
                
                image_data, metadata = method(display_index)
                
                # Basic validation - check if we got data
                if not image_data:
                    raise ScreenshotError("No image data captured")
                
                # Save to file if path provided
                if save_path:
                    self._save_screenshot(image_data, save_path)
                    metadata.file_size = os.path.getsize(save_path)
                
                self.logger.info(
                    "Screenshot captured successfully",
                    method=method.__name__,
                    width=metadata.width,
                    height=metadata.height,
                    file_size=len(image_data),
                )
                
                return image_data, metadata
                
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Capture method {method.__name__} failed",
                    error=str(e),
                    method_index=i,
                )
                continue
        
        # All methods failed
        error_msg = f"All capture methods failed. Last error: {last_error}"
        self.logger.error(error_msg)
        raise ScreenshotError(error_msg, capture_method="all_failed")
    
    def _capture_quartz(self, display_index: int) -> Tuple[bytes, ScreenshotMetadata]:
        """Capture using Quartz/CoreGraphics on macOS"""
        if self.system_info.os_name != "macos":
            raise ScreenshotError("Quartz capture only available on macOS")
        
        try:
            from Quartz import CGDisplayBounds
            from Quartz import CGMainDisplayID
            from Quartz import CGDisplayCreateImage
            from Quartz import CGImageGetBytesPerRow
            from Quartz import CGImageGetWidth
            from Quartz import CGImageGetHeight
            from Quartz import CGImageGetBitsPerPixel
            from Quartz import CGDataProviderCopyData
            from Quartz import CGImageGetDataProvider
            
            # Get display information
            if display_index == 0:
                display_id = CGMainDisplayID()
            else:
                # Get all displays and select by index
                from Quartz import CGGetActiveDisplayList
                displays = CGGetActiveDisplayList(32, None, None)[0]
                if display_index >= len(displays):
                    raise ScreenshotError(f"Display index {display_index} out of range")
                display_id = displays[display_index]
            
            bounds = CGDisplayBounds(display_id)
            width = int(bounds.size.width)
            height = int(bounds.size.height)
            
            # Capture screenshot
            image_ref = CGDisplayCreateImage(display_id)
            if not image_ref:
                raise ScreenshotError("Failed to create image from display")
            
            try:
                # Get image data
                provider = CGImageGetDataProvider(image_ref)
                data = CGDataProviderCopyData(provider)
                
                # Convert to PIL Image
                image = Image.frombytes(
                    "RGBA",
                    (width, height),
                    data,
                    "raw",
                    "BGRA",
                    CGImageGetBytesPerRow(image_ref),
                    1
                )
                
                # Convert to RGB if needed
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Save to bytes
                buffer = io.BytesIO()
                image.save(buffer, format=self.format, quality=self.quality)
                image_data = buffer.getvalue()
                
                # Create metadata
                metadata = ScreenshotMetadata(
                    timestamp=datetime.now(),
                    width=width,
                    height=height,
                    format=self.format,
                    quality=self.quality,
                    file_size=len(image_data),
                    capture_method="quartz",
                    platform=self.system_info.os_name,
                    scale_factor=self.system_info.scale_factor,
                    display_index=display_index,
                    color_depth=CGImageGetBitsPerPixel(image_ref),
                )
                
                return image_data, metadata
                
            finally:
                # Clean up
                import objc
                objc.release(image_ref)
                
        except ImportError as e:
            raise ScreenshotError(f"Quartz not available: {e}", capture_method="quartz")
        except Exception as e:
            raise ScreenshotError(f"Quartz capture failed: {e}", capture_method="quartz")
    
    def _capture_win32(self, display_index: int) -> Tuple[bytes, ScreenshotMetadata]:
        """Capture using Win32 API on Windows"""
        if self.system_info.os_name != "windows":
            raise ScreenshotError("Win32 capture only available on Windows")
        
        try:
            import ctypes
            from ctypes import wintypes
            
            # Define Win32 structures and functions
            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ("biSize", wintypes.DWORD),
                    ("biWidth", wintypes.LONG),
                    ("biHeight", wintypes.LONG),
                    ("biPlanes", wintypes.WORD),
                    ("biBitCount", wintypes.WORD),
                    ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD),
                    ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG),
                    ("biClrUsed", wintypes.DWORD),
                    ("biClrImportant", wintypes.DWORD),
                ]
            
            class BITMAPINFO(ctypes.Structure):
                _fields_ = [
                    ("bmiHeader", BITMAPINFOHEADER),
                    ("bmiColors", wintypes.DWORD * 3),
                ]
            
            # Load libraries
            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32
            
            # Get screen dimensions
            width = user32.GetSystemMetrics(0)  # SM_CXSCREEN
            height = user32.GetSystemMetrics(1)  # SM_CYSCREEN
            
            # Create device contexts
            hdc = user32.GetDC(0)
            hdc_mem = gdi32.CreateCompatibleDC(hdc)
            hbitmap = gdi32.CreateCompatibleBitmap(hdc, width, height)
            hbitmap_old = gdi32.SelectObject(hdc_mem, hbitmap)
            
            try:
                # Copy screen
                gdi32.BitBlt(hdc_mem, 0, 0, width, height, hdc, 0, 0, 0x00CC0020)  # SRCCOPY
                
                # Get bitmap data
                bmi = BITMAPINFO()
                bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
                bmi.bmiHeader.biWidth = width
                bmi.bmiHeader.biHeight = -height  # Negative for top-down
                bmi.bmiHeader.biPlanes = 1
                bmi.bmiHeader.biBitCount = 24
                bmi.bmiHeader.biCompression = 0  # BI_RGB
                
                # Create buffer
                buffer_size = width * height * 3
                buffer = ctypes.create_string_buffer(buffer_size)
                
                # Get bitmap bits
                gdi32.GetDIBits(hdc_mem, hbitmap, 0, height, buffer, ctypes.byref(bmi), 0)
                
                # Convert to PIL Image
                image = Image.frombytes("RGB", (width, height), buffer, "raw", "BGR", 0, 1)
                
                # Save to bytes
                img_buffer = io.BytesIO()
                image.save(img_buffer, format=self.format, quality=self.quality)
                image_data = img_buffer.getvalue()
                
                # Create metadata
                metadata = ScreenshotMetadata(
                    timestamp=datetime.now(),
                    width=width,
                    height=height,
                    format=self.format,
                    quality=self.quality,
                    file_size=len(image_data),
                    capture_method="win32",
                    platform=self.system_info.os_name,
                    scale_factor=self.system_info.scale_factor,
                    display_index=display_index,
                    color_depth=24,
                )
                
                return image_data, metadata
                
            finally:
                # Clean up
                gdi32.SelectObject(hdc_mem, hbitmap_old)
                gdi32.DeleteObject(hbitmap)
                gdi32.DeleteDC(hdc_mem)
                user32.ReleaseDC(0, hdc)
                
        except Exception as e:
            raise ScreenshotError(f"Win32 capture failed: {e}", capture_method="win32")
    
    def _capture_x11(self, display_index: int) -> Tuple[bytes, ScreenshotMetadata]:
        """Capture using X11 on Linux"""
        try:
            import subprocess
            
            # Try using xwd and convert
            with tempfile.NamedTemporaryFile(suffix=".xwd", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Capture using xwd
                result = subprocess.run(
                    ["xwd", "-root", "-silent", "-out", tmp_path],
                    capture_output=True,
                    timeout=10
                )
                
                if result.returncode != 0:
                    raise ScreenshotError(f"xwd failed: {result.stderr.decode()}")
                
                # Convert using ImageMagick if available
                try:
                    result = subprocess.run(
                        ["convert", tmp_path, self.format + ":-"],
                        capture_output=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0:
                        image_data = result.stdout
                        
                        # Get image info
                        image = Image.open(io.BytesIO(image_data))
                        width, height = image.size
                        
                        metadata = ScreenshotMetadata(
                            timestamp=datetime.now(),
                            width=width,
                            height=height,
                            format=self.format,
                            quality=self.quality,
                            file_size=len(image_data),
                            capture_method="x11_imagemagick",
                            platform=self.system_info.os_name,
                            scale_factor=self.system_info.scale_factor,
                            display_index=display_index,
                            color_depth=len(image.getbands()) * 8,
                        )
                        
                        return image_data, metadata
                        
                except (FileNotFoundError, subprocess.SubprocessError):
                    pass
                
                # Fallback: use PIL to read xwd file
                image = Image.open(tmp_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                buffer = io.BytesIO()
                image.save(buffer, format=self.format, quality=self.quality)
                image_data = buffer.getvalue()
                
                metadata = ScreenshotMetadata(
                    timestamp=datetime.now(),
                    width=image.width,
                    height=image.height,
                    format=self.format,
                    quality=self.quality,
                    file_size=len(image_data),
                    capture_method="x11_pil",
                    platform=self.system_info.os_name,
                    scale_factor=self.system_info.scale_factor,
                    display_index=display_index,
                    color_depth=len(image.getbands()) * 8,
                )
                
                return image_data, metadata
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                    
        except Exception as e:
            raise ScreenshotError(f"X11 capture failed: {e}", capture_method="x11")
    
    def _capture_wayland(self, display_index: int) -> Tuple[bytes, ScreenshotMetadata]:
        """Capture using Wayland"""
        try:
            import subprocess
            
            # Try using grim (Wayland screenshot tool)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                result = subprocess.run(
                    ["grim", tmp_path],
                    capture_output=True,
                    timeout=10
                )
                
                if result.returncode != 0:
                    raise ScreenshotError(f"grim failed: {result.stderr.decode()}")
                
                # Read the captured image
                with open(tmp_path, "rb") as f:
                    image_data = f.read()
                
                # Convert to desired format if needed
                if self.format != "PNG":
                    image = Image.open(io.BytesIO(image_data))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    
                    buffer = io.BytesIO()
                    image.save(buffer, format=self.format, quality=self.quality)
                    image_data = buffer.getvalue()
                
                # Get image info
                image = Image.open(io.BytesIO(image_data))
                width, height = image.size
                
                metadata = ScreenshotMetadata(
                    timestamp=datetime.now(),
                    width=width,
                    height=height,
                    format=self.format,
                    quality=self.quality,
                    file_size=len(image_data),
                    capture_method="wayland_grim",
                    platform=self.system_info.os_name,
                    scale_factor=self.system_info.scale_factor,
                    display_index=display_index,
                    color_depth=len(image.getbands()) * 8,
                )
                
                return image_data, metadata
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                    
        except Exception as e:
            raise ScreenshotError(f"Wayland capture failed: {e}", capture_method="wayland")
    
    def _capture_pil(self, display_index: int) -> Tuple[bytes, ScreenshotMetadata]:
        """Capture using PIL ImageGrab"""
        try:
            from PIL import ImageGrab
            
            # Capture screenshot
            image = ImageGrab.grab()
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Save to bytes
            buffer = io.BytesIO()
            image.save(buffer, format=self.format, quality=self.quality)
            image_data = buffer.getvalue()
            
            metadata = ScreenshotMetadata(
                timestamp=datetime.now(),
                width=image.width,
                height=image.height,
                format=self.format,
                quality=self.quality,
                file_size=len(image_data),
                capture_method="pil",
                platform=self.system_info.os_name,
                scale_factor=self.system_info.scale_factor,
                display_index=display_index,
                color_depth=len(image.getbands()) * 8,
            )
            
            return image_data, metadata
            
        except Exception as e:
            raise ScreenshotError(f"PIL capture failed: {e}", capture_method="pil")
    
    def _capture_opencv(self, display_index: int) -> Tuple[bytes, ScreenshotMetadata]:
        """Capture using OpenCV"""
        try:
            # Try different OpenCV methods
            if hasattr(cv2, 'CAP_WINRT') and self.system_info.os_name == "windows":
                # Windows-specific method
                cap = cv2.VideoCapture(0, cv2.CAP_WINRT)
            else:
                # Generic method
                cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                raise ScreenshotError("Failed to open camera/device for screenshot")
            
            try:
                ret, frame = cap.read()
                if not ret:
                    raise ScreenshotError("Failed to capture frame")
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                image = Image.fromarray(frame_rgb)
                
                # Save to bytes
                buffer = io.BytesIO()
                image.save(buffer, format=self.format, quality=self.quality)
                image_data = buffer.getvalue()
                
                metadata = ScreenshotMetadata(
                    timestamp=datetime.now(),
                    width=image.width,
                    height=image.height,
                    format=self.format,
                    quality=self.quality,
                    file_size=len(image_data),
                    capture_method="opencv",
                    platform=self.system_info.os_name,
                    scale_factor=self.system_info.scale_factor,
                    display_index=display_index,
                    color_depth=len(image.getbands()) * 8,
                )
                
                return image_data, metadata
                
            finally:
                cap.release()
                
        except Exception as e:
            raise ScreenshotError(f"OpenCV capture failed: {e}", capture_method="opencv")
    
    def _capture_pyautogui(self, display_index: int) -> Tuple[bytes, ScreenshotMetadata]:
        """Capture using PyAutoGUI"""
        try:
            import pyautogui
            
            # Capture screenshot
            screenshot = pyautogui.screenshot()
            
            if screenshot.mode != "RGB":
                screenshot = screenshot.convert("RGB")
            
            # Save to bytes
            buffer = io.BytesIO()
            screenshot.save(buffer, format=self.format, quality=self.quality)
            image_data = buffer.getvalue()
            
            metadata = ScreenshotMetadata(
                timestamp=datetime.now(),
                width=screenshot.width,
                height=screenshot.height,
                format=self.format,
                quality=self.quality,
                file_size=len(image_data),
                capture_method="pyautogui",
                platform=self.system_info.os_name,
                scale_factor=self.system_info.scale_factor,
                display_index=display_index,
                color_depth=len(screenshot.getbands()) * 8,
            )
            
            return image_data, metadata
            
        except Exception as e:
            raise ScreenshotError(f"PyAutoGUI capture failed: {e}", capture_method="pyautogui")
    
    def _save_screenshot(self, image_data: bytes, save_path: str):
        """Save screenshot to file"""
        try:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "wb") as f:
                f.write(image_data)
                
            self.logger.info(
                "Screenshot saved",
                path=str(path),
                size=len(image_data),
            )
            
        except Exception as e:
            raise ScreenshotError(f"Failed to save screenshot: {e}")
    
    def add_label_to_screenshot(
        self,
        image_data: bytes,
        label: str,
        position: str = "top-left",
        font_size: int = 24,
        color: str = "white",
        background: str = "black"
    ) -> bytes:
        """Add label to screenshot"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGBA if needed
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            
            # Create drawing context
            draw = ImageDraw.Draw(image)
            
            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except (OSError, IOError):
                try:
                    # Try system fonts
                    if self.system_info.os_name == "macos":
                        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
                    elif self.system_info.os_name == "windows":
                        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
                    else:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except (OSError, IOError):
                    font = ImageFont.load_default()
            
            # Calculate text size
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Calculate position
            padding = 10
            if position == "top-left":
                x, y = padding, padding
            elif position == "top-right":
                x, y = image.width - text_width - padding, padding
            elif position == "bottom-left":
                x, y = padding, image.height - text_height - padding
            elif position == "bottom-right":
                x, y = image.width - text_width - padding, image.height - text_height - padding
            elif position == "center":
                x, y = (image.width - text_width) // 2, (image.height - text_height) // 2
            else:
                x, y = padding, padding
            
            # Draw background rectangle
            draw.rectangle(
                [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
                fill=background
            )
            
            # Draw text
            draw.text((x, y), label, font=font, fill=color)
            
            # Convert back to original format
            if image.mode == "RGBA" and self.format != "PNG":
                image = image.convert("RGB")
            
            # Save to bytes
            buffer = io.BytesIO()
            image.save(buffer, format=self.format, quality=self.quality)
            labeled_image_data = buffer.getvalue()
            
            self.logger.info(
                "Label added to screenshot",
                label=label,
                position=position,
                original_size=len(image_data),
                labeled_size=len(labeled_image_data),
            )
            
            return labeled_image_data
            
        except Exception as e:
            raise ScreenshotError(f"Failed to add label to screenshot: {e}")
    
    def get_capture_methods(self) -> List[str]:
        """Get list of available capture methods"""
        return [method.__name__ for method in self._capture_methods]
    
    def test_capture_methods(self) -> Dict[str, bool]:
        """Test all capture methods and return results"""
        results = {}
        
        for method in self._capture_methods:
            try:
                method(0)
                results[method.__name__] = True
                self.logger.info(f"Capture method test passed: {method.__name__}")
            except Exception as e:
                results[method.__name__] = False
                self.logger.warning(f"Capture method test failed: {method.__name__}", error=str(e))
        
        return results
