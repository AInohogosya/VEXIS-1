#!/usr/bin/env python3
"""
Gemini Model Definitions for VEXIS-1.2
Optimized model definitions focusing on Gemini family models
Single source of truth for all model classifications
"""

from typing import Dict, List, Any

# Gemini model families with comprehensive information
GEMINI_MODEL_FAMILIES = {
    "google": {
        "name": "Google",
        "description": "Google's Gemini family models - Advanced multimodal AI",
        "icon": "💎",
        "priority": 1,
        "subfamilies": {
            "gemma2": {
                "name": "Gemma 2",
                "description": "Efficient Gemma 2 models",
                "icon": "💎",
                "models": {
                    "gemma2:2b": {"name": "Gemma 2 2B", "desc": "2B parameters • High-performing • Efficient", "icon": "⚡"},
                    "gemma2:9b": {"name": "Gemma 2 9B", "desc": "9B parameters • High-performing • Efficient", "icon": "🧠"},
                    "gemma2:27b": {"name": "Gemma 2 27B", "desc": "27B parameters • High-performing • Efficient", "icon": "💪"},
                }
            },
            "gemma3": {
                "name": "Gemma 3",
                "description": "Latest generation Gemma models with multimodal capabilities",
                "icon": "🔮",
                "models": {
                    "gemma3:latest": {"name": "Gemma 3 Latest", "desc": "4B parameters • 128K context • Multimodal", "icon": "⭐"},
                    "gemma3:27b": {"name": "Gemma 3 27B", "desc": "27B parameters • 128K context • Multimodal", "icon": "👑"},
                    "gemma3:12b": {"name": "Gemma 3 12B", "desc": "12B parameters • 128K context • Multimodal", "icon": "💪"},
                    "gemma3:4b": {"name": "Gemma 3 4B", "desc": "4B parameters • 128K context • Multimodal", "icon": "🪶"},
                    "gemma3:1b": {"name": "Gemma 3 1B", "desc": "1B parameters • 32K context • Text only", "icon": "⚡"},
                }
            },
            "gemini3": {
                "name": "Gemini 3",
                "description": "Google's flagship Gemini 3 models",
                "icon": "🌟",
                "models": {
                    "gemini-3-pro-preview": {"name": "Gemini 3 Pro", "desc": "Advanced reasoning • Complex tasks • Cloud", "icon": "🧠"},
                    "gemini-3-flash-preview": {"name": "Gemini 3 Flash", "desc": "Speed optimized • Cost-effective • Cloud", "icon": "☁️"},
                }
            }
        }
    }
}

# Predefined models for quick access
PREDEFINED_GEMINI_MODELS = {
    # Local Gemma models
    "gemma2:2b": {"family": "google", "subfamily": "gemma2", "type": "local"},
    "gemma2:9b": {"family": "google", "subfamily": "gemma2", "type": "local"},
    "gemma2:27b": {"family": "google", "subfamily": "gemma2", "type": "local"},
    "gemma3:latest": {"family": "google", "subfamily": "gemma3", "type": "local"},
    "gemma3:27b": {"family": "google", "subfamily": "gemma3", "type": "local"},
    "gemma3:12b": {"family": "google", "subfamily": "gemma3", "type": "local"},
    "gemma3:4b": {"family": "google", "subfamily": "gemma3", "type": "local"},
    "gemma3:1b": {"family": "google", "subfamily": "gemma3", "type": "local"},
    
    # Cloud Gemini models
    "gemini-3-pro-preview": {"family": "google", "subfamily": "gemini3", "type": "cloud"},
    "gemini-3-flash-preview": {"family": "google", "subfamily": "gemini3", "type": "cloud"},
}

# Recommended models by use case
RECOMMENDED_MODELS = {
    "general": {
        "model": "gemma2:2b",
        "reason": "Good balance of performance and efficiency"
    },
    "performance": {
        "model": "gemma3:4b",
        "reason": "Latest generation with multimodal capabilities"
    },
    "lightweight": {
        "model": "gemma3:1b",
        "reason": "Ultra lightweight for resource-constrained environments"
    },
    "cloud": {
        "model": "gemini-3-flash-preview",
        "reason": "Fast and cost-effective cloud model"
    },
    "advanced": {
        "model": "gemini-3-pro-preview",
        "reason": "Advanced reasoning for complex tasks"
    }
}


def get_gemini_families() -> Dict[str, Any]:
    """Get all Gemini model families"""
    return GEMINI_MODEL_FAMILIES


def get_gemini_subfamilies(family: str) -> Dict[str, Any]:
    """Get subfamilies for a specific Gemini family"""
    if family in GEMINI_MODEL_FAMILIES:
        return GEMINI_MODEL_FAMILIES[family]["subfamilies"]
    return {}


def get_gemini_models_in_subfamily(family: str, subfamily: str) -> Dict[str, Any]:
    """Get models in a specific subfamily"""
    subfamilies = get_gemini_subfamilies(family)
    if subfamily in subfamilies:
        return subfamilies[subfamily]["models"]
    return {}


def get_gemini_hierarchy_path(model_name: str) -> List[str]:
    """Get the hierarchy path for a model"""
    if model_name in PREDEFINED_GEMINI_MODELS:
        model_info = PREDEFINED_GEMINI_MODELS[model_name]
        return [model_info["family"], model_info["subfamily"]]
    return []


def get_predefined_gemini_models() -> Dict[str, Any]:
    """Get all predefined Gemini models"""
    return PREDEFINED_GEMINI_MODELS


def is_gemini_model(model_name: str) -> bool:
    """Check if a model is a Gemini model"""
    return model_name in PREDEFINED_GEMINI_MODELS


def get_gemini_model_info(model_name: str) -> Dict[str, Any]:
    """Get detailed information about a Gemini model"""
    if not is_gemini_model(model_name):
        return {}
    
    model_info = PREDEFINED_GEMINI_MODELS[model_name]
    family = model_info["family"]
    subfamily = model_info["subfamily"]
    
    # Get detailed model info from the family structure
    models = get_gemini_models_in_subfamily(family, subfamily)
    if model_name in models:
        return {
            **models[model_name],
            "family": family,
            "subfamily": subfamily,
            "type": model_info["type"]
        }
    
    return model_info


def get_recommended_gemini_model(use_case: str = "general") -> Dict[str, Any]:
    """Get recommended model for a specific use case"""
    if use_case in RECOMMENDED_MODELS:
        recommendation = RECOMMENDED_MODELS[use_case]
        model_info = get_gemini_model_info(recommendation["model"])
        return {
            **model_info,
            "reason": recommendation["reason"]
        }
    return {}


def get_gemini_models_by_type(model_type: str) -> List[str]:
    """Get Gemini models by type (local or cloud)"""
    return [
        model for model, info in PREDEFINED_GEMINI_MODELS.items()
        if info["type"] == model_type
    ]


def get_local_gemini_models() -> List[str]:
    """Get all local Gemini models"""
    return get_gemini_models_by_type("local")


def get_cloud_gemini_models() -> List[str]:
    """Get all cloud Gemini models"""
    return get_gemini_models_by_type("cloud")


# Utility functions for model selection
def get_lightweight_gemini_models() -> List[str]:
    """Get lightweight Gemini models (under 4B parameters)"""
    lightweight_models = []
    for model_name in get_local_gemini_models():
        if any(size in model_name for size in [":1b", ":2b", ":3b"]):
            lightweight_models.append(model_name)
    return lightweight_models


def get_performance_gemini_models() -> List[str]:
    """Get high-performance Gemini models"""
    performance_models = []
    for model_name in get_local_gemini_models():
        if any(size in model_name for size in [":12b", ":27b"]):
            performance_models.append(model_name)
    return performance_models


def validate_gemini_model(model_name: str) -> Dict[str, Any]:
    """Validate a Gemini model name"""
    result = {
        "valid": False,
        "gemini_model": False,
        "type": None,
        "error": None,
        "model_name": model_name
    }
    
    if not model_name:
        result["error"] = "Model name cannot be empty"
        return result
    
    # Check if it's a predefined Gemini model
    if is_gemini_model(model_name):
        result["valid"] = True
        result["gemini_model"] = True
        result["type"] = PREDEFINED_GEMINI_MODELS[model_name]["type"]
        return result
    
    # Check if it matches Gemini naming pattern
    if model_name.startswith(("gemma", "gemini")):
        result["gemini_model"] = True
        result["valid"] = True  # Assume valid for custom Gemini models
        result["type"] = "cloud" if "-cloud" in model_name else "local"
    else:
        result["error"] = f"'{model_name}' is not a recognized Gemini model"
    
    return result


if __name__ == "__main__":
    # Test the model definitions
    print("Gemini Model Families:")
    for family, info in get_gemini_families().items():
        print(f"- {info['name']}: {info['description']}")
    
    print("\nRecommended Models:")
    for use_case, recommendation in RECOMMENDED_MODELS.items():
        model_info = get_recommended_gemini_model(use_case)
        print(f"- {use_case}: {model_info.get('name', 'Unknown')} - {recommendation['reason']}")
    
    print("\nLocal Gemini Models:")
    for model in get_local_gemini_models():
        print(f"- {model}")
    
    print("\nCloud Gemini Models:")
    for model in get_cloud_gemini_models():
        print(f"- {model}")
