#!/usr/bin/env python3
"""
Model Availability Checker for VEXIS-1.2
Checks if required Gemini models are available and provides helpful guidance
"""

import subprocess
import sys
import os
from typing import List, Dict, Optional
from pathlib import Path

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ai_agent.utils.ollama_manager import OllamaManager
from ai_agent.utils.model_definitions import (
    get_gemini_families, get_local_gemini_models, get_cloud_gemini_models,
    get_recommended_gemini_model, is_gemini_model, get_gemini_model_info
)


class ModelChecker:
    """Check availability of Gemini models"""
    
    def __init__(self):
        self.ollama_manager = OllamaManager()
        self.ollama_available = self.ollama_manager.check_ollama_available()
        self.available_models = self.ollama_manager.get_installed_models() if self.ollama_available else []
    
    def check_all_models(self) -> Dict[str, Any]:
        """Check all Gemini models"""
        results = {
            'ollama_available': self.ollama_available,
            'total_models': len(self.available_models),
            'gemini_models': [],
            'recommended_models': [],
            'missing_recommended': [],
            'cloud_access': False,
            'issues': [],
            'suggestions': []
        }
        
        if not self.ollama_available:
            results['issues'].append("Ollama is not installed or not running")
            results['suggestions'].append("Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
            return results
        
        # Check Gemini models
        local_gemini = get_local_gemini_models()
        cloud_gemini = get_cloud_gemini_models()
        
        for model_name in local_gemini:
            validation = self.ollama_manager.validate_gemini_model(model_name)
            model_info = get_gemini_model_info(model_name)
            
            results['gemini_models'].append({
                'name': model_name,
                'display_name': model_info.get('name', model_name),
                'type': 'local',
                'installed': validation['installed'],
                'valid': validation['valid'],
                'description': model_info.get('desc', ''),
                'icon': model_info.get('icon', '📋')
            })
        
        for model_name in cloud_gemini:
            validation = self.ollama_manager.validate_gemini_model(model_name)
            model_info = get_gemini_model_info(model_name)
            
            results['gemini_models'].append({
                'name': model_name,
                'display_name': model_info.get('name', model_name),
                'type': 'cloud',
                'installed': validation['installed'],
                'valid': validation['valid'],
                'description': model_info.get('desc', ''),
                'icon': model_info.get('icon', '📋')
            })
        
        # Check cloud access
        cloud_check = self.ollama_manager.check_cloud_model_access()
        results['cloud_access'] = cloud_check['available']
        
        if not cloud_check['available']:
            if not cloud_check['version_ok']:
                results['issues'].append("Ollama version too old for cloud models")
                results['suggestions'].append("Update Ollama: curl -fsSL https://ollama.com/install.sh | sh")
            if not cloud_check['signed_in']:
                results['issues'].append("Not signed in to Ollama cloud")
                results['suggestions'].append("Sign in: ollama signin")
        
        # Check recommended models
        use_cases = ['general', 'performance', 'lightweight', 'cloud']
        for use_case in use_cases:
            try:
                recommended = get_recommended_gemini_model(use_case)
                if recommended:
                    results['recommended_models'].append({
                        'use_case': use_case,
                        'model': recommended.get('name', ''),
                        'reason': recommended.get('reason', ''),
                        'installed': recommended.get('name', '') in self.available_models
                    })
            except Exception as e:
                results['issues'].append(f"Error checking {use_case} model: {e}")
        
        # Find missing recommended models
        for rec in results['recommended_models']:
            if not rec['installed']:
                results['missing_recommended'].append(rec)
        
        return results
    
    def display_results(self, results: Dict[str, Any]):
        """Display model check results"""
        print("=" * 70)
        print("🤖 VEXIS-1.2 Model Availability Check")
        print("=" * 70)
        
        # Basic status
        print(f"\n📊 Status:")
        print(f"  Ollama Available:     {'✓ Yes' if results['ollama_available'] else '✗ No'}")
        print(f"  Total Models:         {results['total_models']}")
        print(f"  Gemini Models:        {len(results['gemini_models'])}")
        print(f"  Cloud Access:         {'✓ Yes' if results['cloud_access'] else '✗ No'}")
        
        # Gemini models by type
        local_models = [m for m in results['gemini_models'] if m['type'] == 'local']
        cloud_models = [m for m in results['gemini_models'] if m['type'] == 'cloud']
        
        if local_models:
            print(f"\n🏠 Local Gemini Models:")
            for model in local_models:
                status = "✓" if model['installed'] else "○"
                print(f"  {status} {model['icon']} {model['display_name']}")
                print(f"      {model['description']}")
        
        if cloud_models:
            print(f"\n☁️  Cloud Gemini Models:")
            for model in cloud_models:
                status = "✓" if model['installed'] else "○"
                print(f"  {status} {model['icon']} {model['display_name']}")
                print(f"      {model['description']}")
        
        # Recommended models
        if results['recommended_models']:
            print(f"\n⭐ Recommended Models:")
            for rec in results['recommended_models']:
                status = "✓" if rec['installed'] else "○"
                print(f"  {status} {rec['model']} ({rec['use_case']})")
                print(f"      {rec['reason']}")
        
        # Issues and suggestions
        if results['issues']:
            print(f"\n❌ Issues:")
            for issue in results['issues']:
                print(f"  - {issue}")
        
        if results['suggestions']:
            print(f"\n💡 Suggestions:")
            for suggestion in results['suggestions']:
                print(f"  - {suggestion}")
        
        if results['missing_recommended']:
            print(f"\n📥 Missing Recommended Models:")
            for missing in results['missing_recommended']:
                print(f"  - {missing['model']} ({missing['use_case']})")
                print(f"    Reason: {missing['reason']}")
            print(f"\n💡 Install missing models:")
            for missing in results['missing_recommended']:
                print(f"  ollama pull {missing['model']}")
        
        print("\n" + "=" * 70)
        print("🎯 Model check completed!")
        print("=" * 70)
    
    def install_missing_models(self, results: Dict[str, Any], interactive: bool = True):
        """Install missing recommended models"""
        if not results['missing_recommended']:
            print("✓ All recommended models are already installed!")
            return True
        
        if interactive:
            print(f"\n📥 Found {len(results['missing_recommended'])} missing recommended models:")
            for i, missing in enumerate(results['missing_recommended'], 1):
                print(f"  {i}. {missing['model']} ({missing['use_case']})")
            
            response = input("\nInstall missing models? (y/N): ").lower().strip()
            if response != 'y':
                print("Skipping model installation.")
                return True
        
        success_count = 0
        for missing in results['missing_recommended']:
            model_name = missing['model']
            print(f"\n📥 Installing {model_name}...")
            
            result = self.ollama_manager.install_model(model_name)
            if result['success']:
                print(f"✓ Successfully installed {model_name}")
                success_count += 1
            else:
                print(f"✗ Failed to install {model_name}: {result.get('error', 'Unknown error')}")
        
        print(f"\n📊 Installation Summary:")
        print(f"  Successfully installed: {success_count}/{len(results['missing_recommended'])}")
        
        return success_count == len(results['missing_recommended'])


def main():
    install_mode = "--install" in sys.argv
    interactive = "--yes" not in sys.argv  # Default to interactive
    
    checker = ModelChecker()
    results = checker.check_all_models()
    checker.display_results(results)
    
    if install_mode and results['missing_recommended']:
        checker.install_missing_models(results, interactive)
    elif install_mode:
        print("\n✓ All recommended models are already installed!")


if __name__ == "__main__":
    main()
