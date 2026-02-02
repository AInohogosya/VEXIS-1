"""
Vision API Client for AI Agent System
Zero-defect policy: simplified API communication with Ollama Cloud Models only
"""

import base64
import io
import json
import time
from typing import Optional, Dict, Any, List, Union, BinaryIO
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

try:
    import requests
except ImportError:
    raise ImportError("requests is required for Vision API client")

try:
    from PIL import Image
except ImportError:
    raise ImportError("PIL (Pillow) is required for Vision API client")

from ..utils.exceptions import APIError, ValidationError
from ..utils.logger import get_logger
from ..utils.config import load_config


class APIProvider(Enum):
    """Supported API providers - Ollama only"""
    OLLAMA = "ollama"


@dataclass
class APIResponse:
    """API response structure"""
    success: bool
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    latency: Optional[float] = None
    error: Optional[str] = None


@dataclass
class APIRequest:
    """API request structure"""
    prompt: str
    image_data: Optional[bytes] = None
    image_format: str = "PNG"
    max_tokens: int = 1000
    temperature: float = 0.7
    model: Optional[str] = None
    provider: Optional[APIProvider] = None


class VisionAPIClient:
    """Vision API client with Ollama only"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config().api.__dict__
        self.logger = get_logger("vision_api_client")
        
        # Initialize only Ollama provider
        self.ollama_provider = OllamaProvider(self.config)
        self.current_provider = self.ollama_provider
        
        self.logger.info(
            "Vision API client initialized - Ollama only",
            ollama_available=self._test_ollama_availability(),
        )
    
    
    def _test_ollama_availability(self) -> bool:
        """Test if Ollama is available"""
        try:
            import requests
            local_endpoint: str = "http://localhost:11434"
            response = requests.get(f"{local_endpoint}/api/tags", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def analyze_image(self, request: APIRequest) -> APIResponse:
        """Analyze image using Ollama only"""
        start_time = time.time()
        
        # Validate request
        self._validate_request(request)
        
        
        try:
            self.logger.debug(
                "Using Ollama for vision analysis",
                model=request.model or self.ollama_provider.default_model,
            )
            
            response = self.ollama_provider.analyze_image(request)
            
            latency = time.time() - start_time
            response.latency = latency
            
            self.logger.info(
                "Ollama image analysis successful",
                model=response.model,
                tokens_used=response.tokens_used,
                latency=latency,
                cost=response.cost,
            )
            
            return response
            
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Ollama analysis failed: {e}"
            
            self.logger.error(error_msg)
            
            return APIResponse(
                success=False,
                content="",
                model="",
                provider="ollama",
                latency=latency,
                error=error_msg,
            )
    
    def _validate_request(self, request: APIRequest):
        """Validate API request"""
        if not request.prompt:
            raise ValidationError("Prompt cannot be empty", "prompt", request.prompt)
        
        if len(request.prompt) > 10000:
            raise ValidationError("Prompt too long", "prompt", len(request.prompt))
        
        if request.image_data:
            if len(request.image_data) > 20 * 1024 * 1024:  # 20MB limit
                raise ValidationError("Image too large", "image_data", len(request.image_data))
            
            # Validate image format
            try:
                Image.open(io.BytesIO(request.image_data))
            except Exception as e:
                raise ValidationError(f"Invalid image format: {e}", "image_data", "format_error")
        
        if request.max_tokens < 1 or request.max_tokens > 4000:
            raise ValidationError("Invalid max_tokens", "max_tokens", request.max_tokens)
        
        if not (0.0 <= request.temperature <= 2.0):
            raise ValidationError("Invalid temperature", "temperature", request.temperature)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers - Ollama only"""
        providers = []
        if self._test_ollama_availability():
            providers.append("ollama")
        return providers
    
    def get_current_provider(self) -> str:
        """Get current provider name"""
        return self.current_provider.name
    
    def test_providers(self) -> Dict[str, bool]:
        """Test Ollama provider only"""
        results = {}
        
        # Test Ollama
        try:
            test_request = APIRequest(
                prompt="Describe this image briefly.",
                image_data=self._create_test_image(),
                max_tokens=50,
                temperature=0.1,
            )
            
            response = self.ollama_provider.analyze_image(test_request)
            results["ollama"] = response.success
            
            if response.success:
                self.logger.info("Ollama provider test passed")
            else:
                self.logger.warning("Ollama provider test failed", error=response.error)
                
        except Exception as e:
            results["ollama"] = False
            self.logger.warning("Ollama provider test failed", error=str(e))
        
        return results
    
    def _create_test_image(self) -> bytes:
        """Create a simple test image"""
        image = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()
    


class BaseAPIProvider:
    """Base class for API providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"provider_{self.name.lower()}")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
    
    @property
    def name(self) -> str:
        """Provider name"""
        raise NotImplementedError
    
    @property
    def default_model(self) -> str:
        """Default model for this provider"""
        raise NotImplementedError
    
    def analyze_image(self, request: APIRequest) -> APIResponse:
        """Analyze image - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _make_request_with_retry(self, url: str, headers: Dict[str, str], data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    raise APIError(error_msg, status_code=response.status_code)
                    
            except requests.exceptions.Timeout:
                last_error = "Request timeout"
                self.logger.warning(f"Request timeout (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError:
                last_error = "Connection error"
                self.logger.warning(f"Connection error (attempt {attempt + 1})")
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Request failed (attempt {attempt + 1})", error=str(e))
            
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        raise APIError(f"Request failed after {self.max_retries + 1} attempts: {last_error}")
    


class OllamaProvider:
    """Ollama provider for local vision models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("provider_ollama")
        self.timeout = config.get("timeout", 120)  # Increased timeout for vision models
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        self.endpoint = config.get("local_endpoint", "http://localhost:11434")
    
    @property
    def name(self) -> str:
        return "ollama"
    
    @property
    def default_model(self) -> str:
        return self.config.get("local_model", "gemini-3-flash-preview:latest")
    
    def analyze_image(self, request: APIRequest) -> APIResponse:
        """Analyze image using Ollama API (supports both local and cloud models)"""
        try:
            import requests
            import base64
            
            # Prepare the request payload for /api/chat endpoint
            payload = {
                "model": request.model or self.default_model,
                "messages": [{
                    "role": "user",
                    "content": request.prompt,
                }],
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                }
            }
            
            # Add image if provided (for vision models)
            if request.image_data:
                # Convert image to base64
                image_base64 = base64.b64encode(request.image_data).decode('utf-8')
                payload["messages"][0]["images"] = [image_base64]
            
            # Make API call to /api/chat endpoint
            response = requests.post(
                f"{self.endpoint}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                # Extract content from chat response format
                content = result.get("message", {}).get("content", "")
                
                return APIResponse(
                    success=True,
                    content=content,
                    model=request.model or self.default_model,
                    provider=self.name,
                    cost=0.0,  # Cloud models have costs, but handled by Ollama account
                    tokens_used=result.get("eval_count", None),
                )
            else:
                return APIResponse(
                    success=False,
                    content="",
                    model=request.model or self.default_model,
                    provider=self.name,
                    error=f"HTTP {response.status_code}: {response.text}",
                )
            
        except Exception as e:
            return APIResponse(
                success=False,
                content="",
                model=request.model or self.default_model,
                provider=self.name,
                error=str(e),
            )
    
    def _calculate_cost(self, model: str, tokens: Optional[int] = None) -> Optional[float]:
        """Calculate cost for Ollama (always free for local models)"""
        return 0.0
