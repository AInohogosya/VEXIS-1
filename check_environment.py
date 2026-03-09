#!/usr/bin/env python3
"""
Standalone Environment Check Script for VEXIS-1.2
Usage: python3 check_environment.py [--fix]
"""

import sys
import os

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ai_agent.utils.environment_detector import detect_and_plan, AdaptiveExecutor


def main():
    fix_mode = "--fix" in sys.argv
    
    print("=" * 70)
    print("🔍 VEXIS-1.2 Environment Detection and Setup")
    print("=" * 70)
    
    # Run detection and create plan
    env_info, executor = detect_and_plan()
    
    # Show environment status
    print("\n" + "=" * 70)
    print("📊 Environment Status Summary")
    print("=" * 70)
    print(f"  OS System:            {env_info.os_system} {env_info.os_release}")
    print(f"  Python Version:        {env_info.python_version}")
    print(f"  Ollama Available:      {'✓ Yes' if env_info.ollama_available else '✗ No'}")
    print(f"  Ollama Version:        {env_info.ollama_version or 'N/A'}")
    print(f"  Cloud Model Support:   {'✓ Yes' if env_info.can_use_cloud_models else '✗ No'}")
    print(f"  Local Models:          {len(env_info.ollama_local_models)} installed")
    print(f"  Cloud Models:          {len(env_info.ollama_cloud_models)} installed")
    print(f"  Recommended Provider:  {env_info.recommended_provider}")
    
    # Show installed models
    if env_info.ollama_installed_models:
        print(f"\n📦 Installed Models:")
        for model in env_info.ollama_installed_models[:10]:  # Show first 10
            print(f"  - {model}")
        if len(env_info.ollama_installed_models) > 10:
            print(f"  ... and {len(env_info.ollama_installed_models) - 10} more")
    
    # Show warnings
    if env_info.warnings:
        print(f"\n⚠️  Warnings:")
        for warning in env_info.warnings:
            print(f"  - {warning}")
    
    # Show suggestions
    if env_info.suggestions:
        print(f"\n💡 Suggestions:")
        for suggestion in env_info.suggestions:
            print(f"  - {suggestion}")
    
    # Show execution plan
    if executor.execution_plan:
        print(f"\n🔧 Recommended Actions:")
        for i, step in enumerate(executor.execution_plan, 1):
            print(f"  {i}. {step['description']}")
            if step.get('command'):
                print(f"     Command: {step['command']}")
    
    # Execute fix plan if requested
    if fix_mode and executor.execution_plan:
        print(f"\n🔧 Fix mode enabled - executing {len(executor.execution_plan)} steps\n")
        success = executor.execute_plan(interactive=True)
        if success:
            print("\n✓ Setup completed successfully!")
            print("\nYou can now run: python3 run.py \"your command\"")
        else:
            print("\n⚠ Setup completed with some issues.")
            print("Check the error messages above for details.")
    elif executor.execution_plan:
        print(f"\n💡 Run with --fix to automatically address these issues")
    
    # Show network status
    print(f"\n🌐 Network Connectivity:")
    print(f"  Ollama.com:           {'✓ Connected' if env_info.can_connect_to_ollama_com else '✗ Disconnected'}")
    print(f"  PyPI.org:             {'✓ Connected' if env_info.can_connect_to_pypi else '✗ Disconnected'}")
    
    # Show system capabilities
    print(f"\n🛠️  System Capabilities:")
    print(f"  Virtual Environment:   {'✓ Available' if env_info.has_venv_module else '✗ Not Available'}")
    print(f"  Docker:               {'✓ Available' if env_info.has_docker else '✗ Not Available'}")
    print(f"  Git:                  {'✓ Available' if env_info.has_git else '✗ Not Available'}")
    print(f"  Curl:                 {'✓ Available' if env_info.has_curl else '✗ Not Available'}")
    
    print("\n" + "=" * 70)
    print("🎯 Environment check completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
