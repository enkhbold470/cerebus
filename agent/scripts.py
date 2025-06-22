#!/usr/bin/env python3
"""
Poetry scripts for Smart Glasses ReAct AI Agent
Run these with: poetry run <script-name>
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def run_web():
    """Run the agent with ADK web interface"""
    print("🌐 Starting Smart Glasses Agent - Web Interface")
    print("📖 Access at: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        subprocess.run(["adk", "web"], cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("\n🛑 Agent stopped")
    except FileNotFoundError:
        print("❌ ADK not found. Install with: pip install google-adk")
        sys.exit(1)

def run_cli():
    """Run the agent with ADK CLI interface"""
    print("💻 Starting Smart Glasses Agent - CLI Interface")
    print("💬 Type your queries and press Enter")
    print("🛑 Type 'exit' or press Ctrl+C to stop")
    
    try:
        subprocess.run(["adk", "run", "."], cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("\n🛑 Agent stopped")
    except FileNotFoundError:
        print("❌ ADK not found. Install with: pip install google-adk")
        sys.exit(1)

def run_tests():
    """Run the agent test suite"""
    print("🧪 Running Smart Glasses Agent Test Suite")
    
    try:
        test_script = Path(__file__).parent / "test_agent.py"
        subprocess.run([sys.executable, str(test_script)])
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

def setup_environment():
    """Interactive environment setup"""
    print("⚙️ Smart Glasses Agent - Environment Setup")
    print("=" * 50)
    
    env_file = Path(__file__).parent / ".env"
    env_template = Path(__file__).parent / "env_template.txt"
    
    if env_file.exists():
        print("✅ .env file already exists")
        response = input("❓ Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("🛑 Setup cancelled")
            return
    
    if not env_template.exists():
        print("❌ env_template.txt not found")
        sys.exit(1)
    
    # Copy template to .env
    import shutil
    shutil.copy(env_template, env_file)
    print(f"📋 Copied {env_template} to {env_file}")
    
    print("\n🔑 Environment Setup Steps:")
    print("1. Get your API key from Google AI Studio: https://aistudio.google.com/")
    print("2. Edit .env file and replace 'your_gemini_api_key_here' with your actual API key")
    print("3. Configure other settings as needed")
    print("4. Run: poetry run agent-test")
    
    # Ask if they want to open the .env file
    response = input("\n❓ Open .env file for editing now? (y/N): ")
    if response.lower() == 'y':
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", str(env_file)])
            elif sys.platform == "linux":  # Linux
                subprocess.run(["xdg-open", str(env_file)])
            elif sys.platform == "win32":  # Windows
                subprocess.run(["notepad", str(env_file)])
            else:
                print(f"📝 Please edit: {env_file}")
        except Exception:
            print(f"📝 Please manually edit: {env_file}")

def debug_agent():
    """Debug agent configuration and dependencies"""
    print("🐛 Smart Glasses Agent - Debug Mode")
    print("=" * 50)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    required_packages = [
        "google.adk.agents",
        "google.adk.tools", 
        "pydantic",
        "python_dotenv"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
    
    # Check environment
    print("\n🔧 Checking environment...")
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print("✅ .env file exists")
        
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            
            if os.getenv("GOOGLE_API_KEY"):
                key = os.getenv("GOOGLE_API_KEY")
                masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
                print(f"✅ GOOGLE_API_KEY: {masked_key}")
            else:
                print("❌ GOOGLE_API_KEY not set")
        except ImportError:
            print("❌ python-dotenv not installed")
    else:
        print("❌ .env file not found")
    
    # Check agent import
    print("\n🤖 Checking agent import...")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from agent.agent import root_agent
        print("✅ Agent imported successfully")
        print(f"   Model: {root_agent.model}")
        print(f"   Sub-agents: {len(root_agent.sub_agents)}")
        print(f"   Tools: {len(root_agent.tools)}")
    except Exception as e:
        print(f"❌ Agent import failed: {e}")
    
    print("\n💡 If you see errors above:")
    print("   1. Run: poetry run agent-setup")
    print("   2. Run: poetry install")
    print("   3. Check your .env configuration")

# CLI entry points for direct execution
if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "web":
            run_web()
        elif command == "cli":
            run_cli()
        elif command == "test":
            run_tests()
        elif command == "setup":
            setup_environment()
        elif command == "debug":
            debug_agent()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: web, cli, test, setup, debug")
    else:
        print("Available commands: web, cli, test, setup, debug")
        print("Usage: python scripts.py <command>") 