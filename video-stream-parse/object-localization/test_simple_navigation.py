#!/usr/bin/env python3
"""
Test Simple Navigation System
Demonstrates how to use the simplified navigation system
"""

import json
import base64
import requests
from simple_navigation import SimpleNavigation, DIRECTIONS
from PIL import Image, ImageDraw
import tempfile
import os

def create_test_image():
    """Create a test image with some objects"""
    # Create a simple test image
    width, height = 640, 480
    image = Image.new('RGB', (width, height), color=(100, 100, 100))
    draw = ImageDraw.Draw(image)
    
    # Draw some rectangles to simulate objects
    # Person on the right
    draw.rectangle([450, 200, 550, 400], fill=(255, 100, 100))  # Red rectangle
    
    # Chair on the left
    draw.rectangle([100, 250, 180, 350], fill=(100, 255, 100))  # Green rectangle
    
    # Car in center-left
    draw.rectangle([200, 300, 350, 380], fill=(100, 100, 255))  # Blue rectangle
    
    # Save test image
    test_image_path = "test_navigation_image.jpg"
    image.save(test_image_path, "JPEG")
    print(f"✓ Created test image: {test_image_path}")
    
    return test_image_path

def test_navigation_direct():
    """Test navigation system directly"""
    print("\n" + "="*50)
    print("Testing Navigation System Directly")
    print("="*50)
    
    # Create test image
    test_image_path = create_test_image()
    
    try:
        # Initialize navigation system
        nav = SimpleNavigation()
        print("✓ Navigation system initialized")
        
        # Analyze the test image
        result = nav.analyze_image(test_image_path)
        
        print("\nNavigation Analysis Result:")
        print(f"Status: {result['status']}")
        print(f"Direction: {result['direction']}")
        print(f"Instruction: {result['instruction']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        print(f"\nDetected Objects: {len(result['detected_objects'])}")
        for obj in result['detected_objects']:
            print(f"  - {obj['object']}: confidence={obj['confidence']:.2f}, "
                  f"angle={obj['angle']:.1f}°, distance={obj['distance']:.2f}, "
                  f"priority={obj['priority']}")
        
        print(f"\nSafe Zones:")
        for zone in result['safe_zones']:
            recommended = "✓" if zone['recommended'] else "✗"
            print(f"  {recommended} {zone['zone']}: safety_score={zone['safety_score']:.1f}")
        
        if result['warnings']:
            print(f"\nWarnings:")
            for warning in result['warnings']:
                print(f"  ⚠️  {warning}")
        
    except Exception as e:
        print(f"✗ Error testing navigation: {e}")

def test_navigation_api():
    """Test navigation system via API"""
    print("\n" + "="*50)
    print("Testing Navigation API")
    print("="*50)
    
    api_url = "http://localhost:5000"
    
    # Test API status
    try:
        response = requests.get(f"{api_url}/status", timeout=5)
        if response.status_code == 200:
            print("✓ API server is running")
            status = response.json()
            print(f"  Status: {status['status']}")
            print(f"  YOLO Model: {status['yolo_model']}")
        else:
            print(f"✗ API server returned status {response.status_code}")
            return
    except requests.exceptions.RequestException:
        print("✗ API server is not running")
        print("  Start it with: python simple_navigation_api.py")
        return
    
    # Test with sample data
    try:
        response = requests.get(f"{api_url}/test", timeout=5)
        if response.status_code == 200:
            print("✓ Test endpoint working")
            test_data = response.json()
            print(f"  Direction: {test_data['direction']}")
            print(f"  Instruction: {test_data['instruction']}")
        else:
            print(f"✗ Test endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Test endpoint error: {e}")
    
    # Test image analysis
    print("\nTesting image analysis via API...")
    test_image_path = create_test_image()
    
    try:
        # Read and encode image
        with open(test_image_path, 'rb') as f:
            image_data = f.read()
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Send to API
        payload = {
            'image_base64': image_base64,
            'image_format': 'jpg'
        }
        
        response = requests.post(f"{api_url}/analyze", 
                               json=payload, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Image analysis successful")
            print(f"  Direction: {result['direction']}")
            print(f"  Instruction: {result['instruction']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Objects detected: {len(result['detected_objects'])}")
        else:
            print(f"✗ Image analysis failed: {response.status_code}")
            print(f"  Response: {response.text}")
    
    except Exception as e:
        print(f"✗ API test error: {e}")

def test_directions():
    """Test direction constants"""
    print("\n" + "="*50)
    print("Testing Direction Constants")
    print("="*50)
    
    print("Available directions:")
    for key, value in DIRECTIONS.items():
        print(f"  {key}: '{value}'")

def test_vector_math():
    """Test vector mathematics with sample data"""
    print("\n" + "="*50)
    print("Testing Vector Mathematics")
    print("="*50)
    
    # Sample objects
    center_x, center_y = 320, 240
    objects = [
        {"name": "person", "position": (450, 240), "priority": 8},
        {"name": "chair", "position": (150, 240), "priority": 5},
        {"name": "car", "position": (320, 180), "priority": 9}
    ]
    
    print(f"Image center: ({center_x}, {center_y})")
    print("Objects:")
    
    for obj in objects:
        x, y = obj["position"]
        # Calculate vector from center to object
        vector_x = x - center_x
        vector_y = y - center_y
        
        # Calculate magnitude (distance)
        import math
        magnitude = math.sqrt(vector_x**2 + vector_y**2)
        
        # Calculate angle
        angle = math.degrees(math.atan2(vector_y, vector_x))
        
        # Calculate navigation angle (horizontal only)
        nav_angle = math.degrees(math.atan(vector_x / (640 / 2)))
        nav_angle = max(-90, min(90, nav_angle))
        
        print(f"  {obj['name']}:")
        print(f"    Position: {obj['position']}")
        print(f"    Vector: ({vector_x}, {vector_y})")
        print(f"    Magnitude: {magnitude:.1f}")
        print(f"    Full angle: {angle:.1f}°")
        print(f"    Navigation angle: {nav_angle:.1f}°")
        print(f"    Priority: {obj['priority']}")

def main():
    """Run all tests"""
    print("Simple Navigation System Tests")
    print("="*50)
    
    # Test direction constants
    test_directions()
    
    # Test vector math
    test_vector_math()
    
    # Test navigation system directly
    test_navigation_direct()
    
    # Test API (if running)
    test_navigation_api()
    
    print("\n" + "="*50)
    print("Tests completed!")
    print("\nTo start the API server:")
    print("  python simple_navigation_api.py")
    print("\nTo test with your own image:")
    print("  from simple_navigation import SimpleNavigation")
    print("  nav = SimpleNavigation()")
    print("  result = nav.analyze_image('your_image.jpg')")

if __name__ == "__main__":
    main()