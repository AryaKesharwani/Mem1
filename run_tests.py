#!/usr/bin/env python3
"""
Simple test runner for Memory Agent system.
Executes the comprehensive test suite and displays results.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the memory agent tests."""
    print("🚀 Starting Memory Agent Test Suite")
    print("=" * 50)
    
    # Change to the project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Ensure data directory exists
    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Run the test script
        result = subprocess.run([
            sys.executable, "test_memory_agent.py"
        ], capture_output=True, text=True, timeout=300)
        
        # Display output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Display result
        if result.returncode == 0:
            print("\n✅ Tests completed successfully!")
        else:
            print(f"\n❌ Tests failed with return code: {result.returncode}")
            
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out after 5 minutes")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
