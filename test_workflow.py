"""
Workflow test script - Test ValTune-Agent with mock data
"""

import os
import sys
import subprocess
from pathlib import Path


def setup_test_env():
    """Setup test environment"""
    print("=" * 60)
    print("ValTune-Agent Workflow Test")
    print("=" * 60)
    print()
    
    # Check DEEPSEEK_API_KEY
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  Warning: DEEPSEEK_API_KEY not set")
        print("Run: $env:DEEPSEEK_API_KEY='your-key'  (PowerShell)")
        print("Or:  export DEEPSEEK_API_KEY='your-key'  (Bash)")
        print()
        response = input("Run in test mode (without real API)? [Y/n]: ")
        if response.lower() != 'n':
            print("✅ Test mode enabled")
            return True
        else:
            sys.exit(1)
    else:
        print(f"✅ DEEPSEEK_API_KEY configured: {api_key[:10]}...")
        print()
        return False


def run_test(use_simple_loop=False):
    """Run the test"""
    
    # Ensure config exists
    config_path = Path("configs/base.yaml")
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print("Please create the configuration file first")
        return
    
    print("📋 Test Configuration:")
    print(f"  - Config: {config_path}")
    print(f"  - Training command: python mock_train.py --cfg {{config}} --run_dir {{run_dir}}")
    print(f"  - Max rounds: 5")
    print(f"  - Mode: {'Simple Loop (Test)' if use_simple_loop else 'LangGraph (Default)'}")
    print()
    
    # Build command
    cmd = [
        sys.executable,
        "cli.py",
        "--base-config", str(config_path),
        "--train-cmd", "python mock_train.py --cfg {config} --run_dir {run_dir}",
        "--max-rounds", "5",
    ]
    
    if use_simple_loop:
        cmd.append("--simple-loop")
    
    print("🚀 Starting test...")
    print(f"Command: {' '.join(cmd)}")
    print()
    print("=" * 60)
    print()
    
    # Run
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("=" * 60)
        print("✅ Test completed successfully!")
        print()
        print("📂 View Results:")
        print("  - runs/round_0/results.json")
        print("  - runs/round_0/metadata.json")
        print("  - runs/round_0/train_stdout.log")
        print()
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print(f"❌ Test failed: {e}")
        print()
        return False
    except KeyboardInterrupt:
        print()
        print("⚠️  Test interrupted")
        print()
        return False
    
    return True


def main():
    use_simple = setup_test_env()
    
    print("Start test? [Y/n]: ", end='')
    response = input()
    if response.lower() == 'n':
        print("Cancelled")
        return
    
    print()
    success = run_test(use_simple_loop=use_simple)
    
    if success:
        print("🎉 Workflow test successful!")
        print()
        print("💡 Tips:")
        print("  1. Check output files in runs/ directory")
        print("  2. Review metadata.json in each round for LLM decisions")
        print("  3. With real API, examine the reasoning in decision logs")
        print()


if __name__ == "__main__":
    main()
