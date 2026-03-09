#!/usr/bin/env python3
"""
VEXIS-1.2 AI Agent Runner
Concise, modular implementation for AI-powered GUI automation

Usage: python3 run.py "your instruction here"
"""

import sys
import os
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

VENV_RESTART_FLAG = "--__venv_restarted__"

def show_help():
    """Show help message"""
    print("VEXIS-1.2 AI Agent Runner")
    print("=" * 50)
    print("Usage: python3 run.py \"your instruction here\"")
    print()
    print("Features:")
    print("  • Automatic virtual environment management")
    print("  • Dependency installation")
    print("  • Model selection (Ollama/Google API)")
    print("  • Cross-platform compatibility")
    print()
    print("Examples:")
    print("  python3 run.py \"Take a screenshot\"")
    print("  python3 run.py \"Open browser and search for AI\"")
    print()
    print("Options:")
    print("  --help, -h          Show this help")
    print("  --debug             Enable debug mode")
    print("  --check-env, -c     Run environment check")
    print("  --check-models, -m  Check model availability")
    print("  --system-check, -s  Run system check")
    print("  --no-prompt         Use saved configuration")

def bootstrap_environment(project_root: Path) -> bool:
    """Bootstrap the environment - create venv and install dependencies with user consent"""
    # Import here to avoid import errors before venv setup
    from ai_agent.utils.venv_manager import VirtualEnvManager
    from ai_agent.utils.dependency_installer import DependencyInstaller
    
    print("Bootstrapping environment...")
    
    venv_manager = VirtualEnvManager(project_root)
    installer = DependencyInstaller(venv_manager)
    
    if not venv_manager.check_prerequisites():
        print("\nVirtual environment prerequisites not met.")
        print("Install python3-venv package and try again.")
        return False
    
    if not venv_manager.create_environment_with_consent():
        print("Virtual environment setup cancelled or failed.")
        return False
    
    if not installer.install_all_dependencies_with_consent(venv_manager.project_root):
        print("Dependency installation cancelled or failed.")
        return False
    
    print("✓ Environment bootstrap complete")
    return True

def main():
    """Main entry point"""
    project_root = Path(__file__).parent
    
    # Handle help and check flags first
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit(0)
    
    if "--check-env" in sys.argv or "-c" in sys.argv:
        try:
            from ai_agent.utils.environment_detector import detect_and_plan
            env_info, executor = detect_and_plan()
            print("\n📊 Environment Status Summary")
            print("=" * 50)
            print(f"  Ollama Available:     {'✓ Yes' if env_info.ollama_available else '✗ No'}")
            print(f"  Cloud Model Support: {'✓ Yes' if env_info.can_use_cloud_models else '✗ No'}")
            print(f"  Recommended Provider: {env_info.recommended_provider}")
            
            if "--fix" in sys.argv:
                print("\n🔧 Fix mode enabled - executing setup steps")
                executor.execute_plan(interactive=True)
        except ImportError as e:
            print(f"Environment check not available: {e}")
        sys.exit(0)
    
    if "--check-models" in sys.argv or "-m" in sys.argv:
        try:
            from check_models import ModelChecker
            checker = ModelChecker()
            results = checker.check_all_models()
            checker.display_results(results)
            
            if "--install" in sys.argv:
                checker.install_missing_models(results)
        except ImportError as e:
            print(f"Model check not available: {e}")
        sys.exit(0)
    
    if "--system-check" in sys.argv or "-s" in sys.argv:
        try:
            from system_check import SystemChecker
            checker = SystemChecker()
            checker.run_all_checks()
            checker.display_results()
        except ImportError as e:
            print(f"System check not available: {e}")
        sys.exit(0)
    
    # Initialize managers after environment is ready
    try:
        from ai_agent.utils.venv_manager import VirtualEnvManager
        from ai_agent.utils.dependency_installer import DependencyInstaller
        from ai_agent.utils.config_manager import ConfigManager
        
        venv_manager = VirtualEnvManager(project_root)
        installer = DependencyInstaller(venv_manager)
        config_manager = ConfigManager()
    except ImportError as e:
        print(f"Failed to import required modules: {e}")
        print("Running environment bootstrap...")
        if not bootstrap_environment(project_root):
            print("Failed to bootstrap environment")
            sys.exit(1)
        # After successful bootstrap, restart in venv
        from ai_agent.utils.venv_manager import VirtualEnvManager
        venv_manager = VirtualEnvManager(project_root)
        print("Restarting in new virtual environment...")
        venv_manager.restart_in_venv(sys.argv[1:], str(__file__))
        return
    
    # Handle virtual environment
    if VENV_RESTART_FLAG in sys.argv:
        sys.argv.remove(VENV_RESTART_FLAG)
        print("✓ Running in virtual environment")
    else:
        if not venv_manager.is_in_virtual_environment():
            venv_python = venv_manager.get_venv_python_path()
            if venv_python:
                try:
                    import subprocess
                    result = subprocess.run([venv_python, "--version"], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        print("Virtual environment found, restarting...")
                        venv_manager.restart_in_venv(sys.argv[1:], str(__file__))
                        return
                except Exception:
                    pass
            
            if not bootstrap_environment(project_root):
                print("Failed to bootstrap environment")
                sys.exit(1)
            
            print("Restarting in new virtual environment...")
            venv_manager.restart_in_venv(sys.argv[1:], str(__file__))
            return
        else:
            print("✓ Already in virtual environment")
    
    # Add navigation module to path
    nav_dir = project_root / "yellow-highlight-navigation"
    if nav_dir.exists():
        sys.path.insert(0, str(nav_dir))
    
    # Validate arguments
    instruction_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    instruction = " ".join(instruction_args)
    
    if not instruction:
        print("Usage: python3 run.py \"your instruction here\"")
        print("Example: python3 run.py \"Take a screenshot\"")
        print("Use --help for more options")
        sys.exit(1)
    
    debug_mode = "--debug" in sys.argv
    
    # Configure provider
    if "--no-prompt" not in sys.argv:
        selected_provider = config_manager.select_provider()
        if selected_provider:
            config_manager.show_config_summary(selected_provider)
    else:
        selected_provider = config_manager.settings_manager.get_preferred_provider()
        print(f"\nUsing saved provider preference: {selected_provider}")
    
    print(f"\nAI Agent executing: {instruction}")
    
    # Run the agent
    try:
        from ai_agent.user_interface.two_phase_app import TwoPhaseAIAgent
        
        config_path = project_root / "config.yaml"
        agent = TwoPhaseAIAgent(config_path=str(config_path) if config_path.exists() else None)
        
        # Update configuration with selected provider
        if hasattr(agent, 'engine') and hasattr(agent.engine, 'model_runner'):
            model_runner = agent.engine.model_runner
            if hasattr(model_runner, 'vision_client'):
                updated_config = model_runner.config.copy()
                updated_config['preferred_provider'] = selected_provider
                updated_config['google_api_key'] = config_manager.settings_manager.get_google_api_key()
                updated_config['google_model'] = config_manager.settings_manager.get_google_model()
                model_runner.vision_client.config = updated_config
        
        result = agent.run(instruction, {"debug": debug_mode})
        
        if result:
            print("\n✓ Task completed successfully")
        else:
            print("\n✗ Task failed")
            sys.exit(1)
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("This suggests a dependency issue. Try deleting the 'venv' directory and running again.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if debug_mode:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
