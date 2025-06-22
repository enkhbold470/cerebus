#!/usr/bin/env python3
"""
Test script for Smart Glasses ReAct AI Agent
Run this to verify your agent setup is working correctly
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_environment():
    """Test that required environment variables are set"""
    print("🔧 Testing Environment Configuration...")

    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("💡 Copy env_template.txt to .env and configure your settings")
        return False
    else:
        print("✅ Environment configuration looks good!")
        return True


def test_agent_import():
    """Test that the agent can be imported successfully"""
    print("\n📦 Testing Agent Import...")

    try:
        from agent.agent import (
            root_agent,
            image_analysis_agent,
            navigation_agent,
            audio_response_agent,
        )

        print("✅ Successfully imported all agents!")
        print(f"   - Root Agent: {root_agent.name}")
        print(f"   - Image Analysis Agent: {image_analysis_agent.name}")
        print(f"   - Navigation Agent: {navigation_agent.name}")
        print(f"   - Audio Response Agent: {audio_response_agent.name}")
        return True
    except Exception as e:
        print(f"❌ Failed to import agents: {e}")
        return False


async def test_agent_basic_functionality():
    """Test basic agent functionality"""
    print("\n🤖 Testing Basic Agent Functionality...")

    try:
        from agent.agent import root_agent

        # Test a simple query
        test_query = "I'm walking and need navigation help. What should I do?"
        print(f"📝 Test Query: {test_query}")

        # This would normally process through the agent
        # For now, just verify the agent is configured correctly
        print(f"✅ Agent is configured with model: {root_agent.model}")
        print(f"✅ Agent has {len(root_agent.sub_agents)} sub-agents")
        print(f"✅ Agent has {len(root_agent.tools)} tools")

        return True
    except Exception as e:
        print(f"❌ Agent functionality test failed: {e}")
        return False


def test_dependencies():
    """Test that required dependencies are installed"""
    print("\n📚 Testing Dependencies...")

    required_packages = [
        "google.adk.agents",
        "google.adk.tools",
        "pydantic",
        "python-dotenv",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")

    if missing_packages:
        print(f"\n💡 Install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies are installed!")
        return True


async def run_integration_test():
    """Run a full integration test simulating the smart glasses workflow"""
    print("\n🔄 Running Integration Test...")

    try:
        from agent.agent import (
            analyze_with_gemini_vision,
            detect_objects_yolo,
            segment_image_segnet,
            process_gps_data,
            generate_navigation_audio,
        )

        # Simulate the workflow
        print("📸 Simulating visual analysis...")
        scene_result = analyze_with_gemini_vision("Urban street scene")
        print(f"   Gemini Vision: {scene_result}")

        objects = detect_objects_yolo("Street scene")
        print(f"   YOLO Detection: {objects}")

        segmentation = segment_image_segnet("Road ahead")
        print(f"   SegNet Result: {segmentation}")

        print("🗺️ Simulating GPS processing...")
        gps_result = process_gps_data("37.7749,-122.4194")
        print(f"   GPS Processing: {gps_result}")

        print("🔊 Simulating audio response...")
        audio_result = generate_navigation_audio("Continue straight ahead", "normal")
        print(f"   Audio Response: {audio_result}")

        print("✅ Integration test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user"""
    print("\n🚀 Next Steps:")
    print("1. Copy env_template.txt to .env and add your GOOGLE_API_KEY")
    print("2. Run 'adk web' to test the agent in the browser")
    print("3. Try these sample queries:")
    print("   - 'What do you see in front of me?'")
    print("   - 'Guide me to the nearest coffee shop'")
    print("   - 'Is it safe to cross the street?'")
    print("4. Integrate the placeholder functions with your actual systems:")
    print("   - Connect vision processing pipeline")
    print("   - Connect GPS handler")
    print("   - Connect audio response system")


async def main():
    """Main test function"""
    print("🧪 Smart Glasses ReAct AI Agent - Test Suite")
    print("=" * 50)

    tests = [
        test_environment(),
        test_dependencies(),
        test_agent_import(),
        await test_agent_basic_functionality(),
        await run_integration_test(),
    ]

    passed = sum(tests)
    total = len(tests)

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Your agent setup is ready.")
        print_next_steps()
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("💡 Make sure you have:")
        print("   - Installed all dependencies: pip install -r requirements.txt")
        print("   - Configured .env file with your API keys")
        print("   - Google AI/Vertex AI access set up")


if __name__ == "__main__":
    asyncio.run(main())
