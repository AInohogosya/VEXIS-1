"""
Google API Provider for VEXIS-1 AI Agent
Handles communication with Google Gemini API
"""

import base64
import io
import json
import time
import uuid
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    import requests
except ImportError:
    raise ImportError("requests is required for Google API provider")

try:
    from PIL import Image
except ImportError:
    raise ImportError("PIL (Pillow) is required for Google API provider")

from .vision_api_client import APIRequest, APIResponse, BaseAPIProvider
from ..utils.exceptions import APIError, ValidationError
from ..utils.logger import get_logger


class GoogleProvider(BaseAPIProvider):
    """Google Gemini API provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("google_api_key")
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent"
        
    @property
    def name(self) -> str:
        return "google"
    
    @property
    def default_model(self) -> str:
        return "gemini-3-flash-preview"
    
    def analyze_image(self, request: APIRequest) -> APIResponse:
        """Analyze image using Google Gemini API"""
        if not self.api_key:
            return APIResponse(
                success=False,
                content="",
                model=self.default_model,
                provider=self.name,
                error="Google API key not configured"
            )
        
        try:
            # Prepare the request payload for Google Gemini API
            # Add unique identifier at the beginning to bypass caching
            unique_id = str(uuid.uuid4())
            unique_prompt = f"Unique Request ID: {unique_id}\n\n{request.prompt}"
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": unique_prompt}
                    ]
                }],
                "generationConfig": {
                    "temperature": request.temperature,
                    "maxOutputTokens": request.max_tokens,
                }
            }
            
            # Add image if provided
            if request.image_data:
                # Convert image to base64
                image_base64 = base64.b64encode(request.image_data).decode('utf-8')
                # Determine MIME type
                image_format = request.image_format.lower()
                mime_type = f"image/{image_format}" if image_format in ["jpeg", "jpg", "png", "webp", "gif"] else "image/png"
                
                payload["contents"][0]["parts"].append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_base64
                    }
                })
            
            # Make API call
            url = f"{self.endpoint}?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                # Extract content from Gemini response format
                content = ""
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        for part in parts:
                            if "text" in part:
                                content += part["text"]
                
                # Extract token usage if available
                tokens_used = None
                if "usageMetadata" in result:
                    tokens_used = result["usageMetadata"].get("totalTokenCount")
                
                return APIResponse(
                    success=True,
                    content=content,
                    model=request.model or self.default_model,
                    provider=self.name,
                    cost=self._calculate_cost(request.model or self.default_model, tokens_used),
                    tokens_used=tokens_used,
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                return APIResponse(
                    success=False,
                    content="",
                    model=request.model or self.default_model,
                    provider=self.name,
                    error=error_msg,
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
        """Calculate cost for Google Gemini API"""
        # Pricing for Gemini 1.5 Flash (as of 2024)
        # These are approximate costs and may change
        if model.startswith("gemini-3-flash-preview"):
            if tokens:
                # Input: $0.075 per 1M tokens, Output: $0.15 per 1M tokens
                # Assuming 50% input, 50% output for estimation
                estimated_cost = (tokens * 0.0001125) / 1000000  # Average cost
                return round(estimated_cost, 6)
        return None
