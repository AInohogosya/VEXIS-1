#!/usr/bin/env python3
"""
Simple AI Agent Runner with Dependency Checking
Usage: python3 run.py "your instruction here"
"""

import sys
import os
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Import dependency checker
try:
    from ai_agent.utils.dependency_checker import check_dependencies
except ImportError:
    print("❌ Cannot import dependency checker. This should not happen.")
    print("Please ensure the src/ai_agent/utils directory exists.")
    sys.exit(1)

def main():
    # Check for help flag first
    if "--help" in sys.argv or "-h" in sys.argv:
        print("VEXIS-1 AI Agent Runner")
        print("=" * 50)
        print("Usage: python3 run.py \"your instruction here\" [options]")
        print()
        print("Examples:")
        print("  python3 run.py \"Take a screenshot\"")
        print("  python3 run.py \"Open a web browser and search for AI\" --debug")
        print()
        print("Options:")
        print("  --help, -h          Show this help message")
        print("  --no-deps-check     Skip dependency checking (not recommended)")
        print("  --debug             Enable debug mode")
        print()
        print("The dependency checker will automatically install missing packages.")
        sys.exit(0)
    
    if len(sys.argv) < 2:
        print("Usage: python3 run.py \"your instruction here\"")
        print("Example: python3 run.py \"Take a screenshot\"")
        print("Use --help for more options")
        sys.exit(1)
    
    # Check for command line flags
    skip_deps_check = "--no-deps-check" in sys.argv
    debug_mode = "--debug" in sys.argv
    
    # Filter out flags to get the actual instruction
    instruction_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    instruction = " ".join(instruction_args)
    
    if not instruction:
        print("❌ No instruction provided")
        print("Usage: python3 run.py \"your instruction here\"")
        sys.exit(1)
    
    print(f"🤖 AI Agent executing: {instruction}")
    
    # Run dependency check unless explicitly skipped
    if not skip_deps_check:
        print("\n🔍 Checking dependencies...")
        deps_ok = check_dependencies(current_dir, auto_install=True)
        
        if not deps_ok:
            print("\n❌ Dependency check failed. Please install missing dependencies manually:")
            print("  pip install -r requirements.txt")
            print("  pip install -e .")
            print("\nOr run with --no-deps-check to skip (not recommended)")
            sys.exit(1)
        
        print("\n✅ Dependencies verified. Starting AI Agent...\n")
    else:
        print("\n⚠️  Skipping dependency check (not recommended)")
    
    try:
        from ai_agent.user_interface.two_phase_app import TwoPhaseAIAgent
        
        # Initialize the agent
        config_path = current_dir / "config.yaml"
        agent = TwoPhaseAIAgent(config_path=str(config_path) if config_path.exists() else None)
        
        # Run the instruction
        options = {"debug": debug_mode}
        result = agent.run(instruction, options)
        
        if result:
            print("✅ Task completed successfully")
        else:
            print("❌ Task failed")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This suggests a dependency issue. Try running without --no-deps-check")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
