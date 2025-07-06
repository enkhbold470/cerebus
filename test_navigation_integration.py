#!/usr/bin/env python3
"""
Test script for GPS/Navigation integration between frontend, server, and agent.

This script tests the complete data flow:
1. Frontend GPS/Maps data → Server endpoints
2. Server data storage and management
3. Agent tools accessing server data

Usage:
    python test_navigation_integration.py
"""

import requests
import json
import time
from typing import Dict, Any


class NavigationIntegrationTester:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.session_id = f"test_{int(time.time())}"

    def test_server_health(self) -> bool:
        """Test if the server is running and healthy."""
        try:
            response = requests.get(f"{self.server_url}/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                print(f"✅ Server is healthy: {status['status']}")
                print(
                    f"   Active connections: {status['active_websocket_connections']}"
                )
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False

    def test_gps_data_push(self) -> bool:
        """Test pushing GPS location data to server."""
        print("\n🧪 Testing GPS location data push...")

        # Simulate GPS data from frontend
        gps_data = {
            "sessionId": self.session_id,
            "lat": 40.7128,
            "lng": -74.0060,
            "accuracy": 10,
            "timestamp": time.time(),
            "speed": 1.5,
            "heading": 45,
        }

        try:
            response = requests.post(
                f"{self.server_url}/gps/location", json=gps_data, timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ GPS data pushed successfully: {result['message']}")
                print(
                    f"   Location: {result['location']['lat']:.6f}, {result['location']['lng']:.6f}"
                )
                return True
            else:
                print(f"❌ GPS push failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error pushing GPS data: {e}")
            return False

    def test_navigation_route_push(self) -> bool:
        """Test pushing navigation route data to server."""
        print("\n🧪 Testing navigation route data push...")

        # Simulate route data from frontend Maps service
        route_data = {
            "sessionId": self.session_id,
            "destination_name": "Central Park",
            "origin": {"lat": 40.7128, "lng": -74.0060},
            "destination": {"lat": 40.7829, "lng": -73.9654},
            "route_summary": {
                "distance": "0.8 mi",
                "duration": "16 mins",
                "travel_mode": "WALKING",
            },
            "detailed_steps": [
                {
                    "instruction": "Head north on Broadway",
                    "distance": "0.2 mi",
                    "duration": "4 mins",
                    "start_location": {"lat": 40.7128, "lng": -74.0060},
                    "end_location": {"lat": 40.7200, "lng": -74.0060},
                },
                {
                    "instruction": "Turn right onto W 34th St",
                    "distance": "0.3 mi",
                    "duration": "6 mins",
                    "start_location": {"lat": 40.7200, "lng": -74.0060},
                    "end_location": {"lat": 40.7200, "lng": -73.9900},
                },
                {
                    "instruction": "Continue to Central Park",
                    "distance": "0.3 mi",
                    "duration": "6 mins",
                    "start_location": {"lat": 40.7200, "lng": -73.9900},
                    "end_location": {"lat": 40.7829, "lng": -73.9654},
                },
            ],
        }

        try:
            response = requests.post(
                f"{self.server_url}/navigation/route", json=route_data, timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Route data pushed successfully: {result['message']}")
                print(f"   Destination: {result['destination']}")
                print(f"   Steps: {result['steps_count']}")
                print(f"   Distance: {result['total_distance']}")
                return True
            else:
                print(f"❌ Route push failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error pushing route data: {e}")
            return False

    def test_agent_navigation_access(self) -> bool:
        """Test agent's ability to access navigation data."""
        print("\n🧪 Testing agent navigation data access...")

        try:
            # Test current navigation status
            response = requests.get(
                f"{self.server_url}/navigation/current?session_id={self.session_id}",
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("has_navigation"):
                    nav_status = data["navigation_status"]
                    print(f"✅ Navigation data accessible to agent:")
                    print(
                        f"   Current step: {nav_status['current_step']}/{nav_status['total_steps']}"
                    )
                    print(
                        f"   Current instruction: {nav_status['current_instruction']}"
                    )
                    print(f"   Remaining steps: {nav_status['remaining_steps']}")

                    # Test next step functionality
                    next_response = requests.get(
                        f"{self.server_url}/navigation/next_step?session_id={self.session_id}",
                        timeout=10,
                    )

                    if next_response.status_code == 200:
                        next_data = next_response.json()
                        if next_data.get("has_step"):
                            print(f"✅ Next step retrieved:")
                            print(
                                f"   Step {next_data['step_number']}: {next_data['instruction']}"
                            )
                            print(
                                f"   Distance: {next_data['distance']}, Duration: {next_data['duration']}"
                            )
                        else:
                            print(
                                f"ℹ️ Navigation complete: {next_data.get('message', 'No more steps')}"
                            )

                    return True
                else:
                    print(
                        f"❌ No navigation data found for session: {data.get('message')}"
                    )
                    return False
            else:
                print(f"❌ Navigation access failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Error accessing navigation data: {e}")
            return False

    def test_gps_location_access(self) -> bool:
        """Test agent's ability to access GPS location data."""
        print("\n🧪 Testing agent GPS location access...")

        try:
            response = requests.get(
                f"{self.server_url}/gps/location/{self.session_id}", timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    location = data["location"]
                    print(f"✅ GPS location accessible to agent:")
                    print(
                        f"   Coordinates: {location['lat']:.6f}, {location['lng']:.6f}"
                    )
                    print(f"   Accuracy: ±{location.get('accuracy', 'unknown')}m")
                    print(f"   Age: {data['age_seconds']:.1f} seconds")
                    return True
                else:
                    print(f"❌ No GPS data found: {data.get('message')}")
                    return False
            else:
                print(f"❌ GPS access failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Error accessing GPS data: {e}")
            return False

    def test_session_management(self) -> bool:
        """Test multi-session management capabilities."""
        print("\n🧪 Testing session management...")

        try:
            response = requests.get(
                f"{self.server_url}/navigation/sessions", timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Session management working:")
                print(f"   Active sessions: {data['active_sessions']}")

                for session in data["sessions"]:
                    print(f"   Session {session['session_id']}:")
                    print(
                        f"     → Destination: {session.get('destination_name', 'Unknown')}"
                    )
                    print(
                        f"     → Progress: {session['current_step']}/{session['total_steps']}"
                    )
                    print(f"     → Mode: {session.get('travel_mode', 'Unknown')}")

                return True
            else:
                print(f"❌ Session management failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Error testing session management: {e}")
            return False

    def run_full_test(self) -> bool:
        """Run the complete integration test suite."""
        print("🚀 Starting GPS/Navigation Integration Test Suite")
        print(f"   Server: {self.server_url}")
        print(f"   Test Session: {self.session_id}")
        print("=" * 60)

        tests = [
            ("Server Health", self.test_server_health),
            ("GPS Data Push", self.test_gps_data_push),
            ("Navigation Route Push", self.test_navigation_route_push),
            ("Agent Navigation Access", self.test_agent_navigation_access),
            ("Agent GPS Access", self.test_gps_location_access),
            ("Session Management", self.test_session_management),
        ]

        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
                if not result:
                    print(f"❌ Test '{test_name}' failed!")
            except Exception as e:
                print(f"💥 Test '{test_name}' crashed: {e}")
                results.append((test_name, False))

        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)

        passed = sum(1 for _, result in results if result)
        total = len(results)

        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")

        print(f"\n🎯 Overall: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All tests passed! Integration is working correctly.")
            print("\n💡 Next steps:")
            print("   1. Start the frontend (cerebus-web) and test GPS collection")
            print(
                "   2. Calculate routes in the frontend and verify server receives data"
            )
            print(
                "   3. Use agent tools in your agent code to access real navigation data"
            )
            return True
        else:
            print("⚠️ Some tests failed. Check server configuration and endpoints.")
            return False


if __name__ == "__main__":
    tester = NavigationIntegrationTester()
    success = tester.run_full_test()
    exit(0 if success else 1)
