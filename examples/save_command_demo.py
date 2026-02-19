#!/usr/bin/env python3
"""
Save Command System Demonstration
Shows how the save() command works in the Phase 2 Execution Engine
"""

import sys
import os
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_agent.core_processing.save_command import SaveCommand, SaveContentType, save


def demo_save_command():
    """Demonstrate the save command system"""
    print("=== Save Command System Demo ===\n")
    
    # Create a save command instance
    save_cmd = SaveCommand(session_id="demo_session")
    
    print("1. Basic save command usage:")
    save("Clicked the browser's search bar. Response received.", 
         operation_command="click(0.5, 0.2)",
         visual_feedback="Search bar highlighted",
         content_type="feedback")
    print("   ✓ Saved basic feedback\n")
    
    print("2. Saving extracted information:")
    save("Extracted filename from dialog: 'document.pdf'", 
         operation_command="extract_text()",
         extracted_info={"filename": "document.pdf", "file_type": "pdf"},
         content_type="extraction")
    print("   ✓ Saved extracted information\n")
    
    print("3. Saving failure information:")
    save("Clicked but no change on screen (no response)", 
         operation_command="click(0.3, 0.7)",
         coordinates=(0.3, 0.7),
         failure_details={"error": "No UI response", "timeout": 5.0},
         content_type="failure")
    print("   ✓ Saved failure information\n")
    
    print("4. Reflection capabilities:")
    previous_content = save_cmd.get_previous_save_content()
    print(f"   Previous save content: {previous_content}")
    
    failure_coords = save_cmd.get_failure_coordinates()
    print(f"   Failure coordinates: {failure_coords}")
    
    extracted_info = save_cmd.get_extracted_information()
    print(f"   Extracted information: {extracted_info}")
    
    print()
    
    print("5. Work log summary:")
    recent_saves = save_cmd.get_recent_saves(5)
    for i, entry in enumerate(recent_saves, 1):
        print(f"   Entry {i}: {entry.content_type.value} - {entry.content[:50]}...")
    
    print("\n6. Session management:")
    save_cmd.end_session()
    print("   ✓ Session ended and work log saved to disk")
    
    print(f"\n=== Demo Complete ===")
    print(f"Work log saved to: ./work_logs/demo_session.json")


def demo_integration_example():
    """Show how save command integrates with automation"""
    print("\n=== Integration Example ===\n")
    
    print("Example automation sequence with save commands:")
    print()
    
    # Simulate automation commands with save integration
    commands = [
        ("click(0.5, 0.2)", "Click browser search bar"),
        ("type_text('Hello World')", "Type search query"),
        ("click(0.8, 0.2)", "Click search button"),
        ("click(0.3, 0.7)", "Click results (failed)"),
        ("click(0.4, 0.7)", "Click results (retry)"),
    ]
    
    for i, (command, description) in enumerate(commands, 1):
        print(f"Step {i}: {description}")
        print(f"   Command: {command}")
        
        # Simulate execution result
        if "failed" in description:
            save(f"Command failed: {command}", 
                 operation_command=command,
                 coordinates=(0.3, 0.7),
                 failure_details={"error": "Element not clickable"},
                 content_type="failure")
            print(f"   Result: ❌ Failed (saved to work log)")
        else:
            save(f"Command succeeded: {command}", 
                 operation_command=command,
                 visual_feedback=f"UI responded to {command}",
                 content_type="feedback")
            print(f"   Result: ✅ Success (saved to work log)")
        
        time.sleep(0.5)  # Simulate delay
    
    print("\n=== Integration Example Complete ===")


if __name__ == "__main__":
    demo_save_command()
    demo_integration_example()
