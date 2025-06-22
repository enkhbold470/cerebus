#!/usr/bin/env python3
"""
Simple Navigation Demo
Shows the core navigation functionality without external dependencies
"""

import json
from simple_navigation import SimpleNavigation, DIRECTIONS
from PIL import Image, ImageDraw

def create_demo_image():
    """Create a demo image with geometric shapes"""
    print("Creating demo image...")
    
    # Create a simple test image
    width, height = 640, 480
    image = Image.new('RGB', (width, height), color=(50, 50, 50))
    draw = ImageDraw.Draw(image)
    
    # Draw some rectangles to simulate objects for YOLO
    # These won't be detected by YOLO, but we can simulate the process
    
    # Person on the right (red rectangle)
    draw.rectangle([450, 200, 550, 400], fill=(255, 100, 100))
    draw.text((460, 180), "Person", fill=(255, 255, 255))
    
    # Chair on the left (green rectangle)  
    draw.rectangle([100, 250, 180, 350], fill=(100, 255, 100))
    draw.text((110, 230), "Chair", fill=(255, 255, 255))
    
    # Car in center-left (blue rectangle)
    draw.rectangle([200, 300, 350, 380], fill=(100, 100, 255))
    draw.text((210, 280), "Car", fill=(255, 255, 255))
    
    # Draw center line for reference
    draw.line([(width//2, 0), (width//2, height)], fill=(128, 128, 128), width=2)
    
    # Save demo image
    demo_image_path = "/Users/inky/Desktop/cerebus/video-stream-parse/demo_output.jpeg"
    image.save(demo_image_path, "JPEG")
    print(f"✓ Demo image saved: {demo_image_path}")
    
    return demo_image_path

def demo_navigation_directions():
    """Demonstrate navigation direction constants"""
    print("\n" + "="*50)
    print("Navigation Directions Available")
    print("="*50)
    
    for key, value in DIRECTIONS.items():
        print(f"  {key:<15} → '{value}'")

def demo_vector_calculations():
    """Demonstrate vector calculations without YOLO"""
    print("\n" + "="*50)
    print("Vector Calculations Demo")
    print("="*50)
    
    # Image dimensions
    image_width, image_height = 640, 480
    center_x, center_y = image_width // 2, image_height // 2
    
    print(f"Image size: {image_width}x{image_height}")
    print(f"Center point: ({center_x}, {center_y})")
    
    # Sample object positions
    objects = [
        {"name": "person", "bbox": [450, 200, 550, 400], "priority": 8},
        {"name": "chair", "bbox": [100, 250, 180, 350], "priority": 5},
        {"name": "car", "bbox": [200, 300, 350, 380], "priority": 9}
    ]
    
    print(f"\nAnalyzing {len(objects)} objects:")
    
    import math
    
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        
        # Calculate object center
        obj_center_x = (x1 + x2) / 2
        obj_center_y = (y1 + y2) / 2
        
        # Calculate angle from image center (-90 to +90)
        rel_x = obj_center_x - center_x
        angle = math.degrees(math.atan(rel_x / (image_width / 2)))
        angle = max(-90, min(90, angle))
        
        # Calculate distance based on size
        width = x2 - x1
        height = y2 - y1
        area = width * height
        image_area = image_width * image_height
        relative_area = area / image_area
        
        if relative_area > 0.3:
            distance = 0.2  # Very close
        elif relative_area > 0.15:
            distance = 0.4  # Close  
        elif relative_area > 0.05:
            distance = 0.6  # Medium
        else:
            distance = 0.8  # Far
        
        print(f"\n  {obj['name'].upper()}:")
        print(f"    Bounding box: [{x1}, {y1}, {x2}, {y2}]")
        print(f"    Center: ({obj_center_x:.1f}, {obj_center_y:.1f})")
        print(f"    Angle from center: {angle:.1f}°")
        print(f"    Estimated distance: {distance:.1f}")
        print(f"    Navigation priority: {obj['priority']}")
        print(f"    Relative area: {relative_area:.3f}")

def demo_direction_calculation():
    """Demonstrate how direction is calculated from obstacles"""
    print("\n" + "="*50)
    print("Direction Calculation Demo")
    print("="*50)
    
    import math
    
    # Sample obstacles with angles
    obstacles = [
        {"name": "person", "angle": 35.0, "priority": 8, "distance": 0.4},
        {"name": "chair", "angle": -25.0, "priority": 5, "distance": 0.6},
        {"name": "car", "angle": -10.0, "priority": 9, "distance": 0.3}
    ]
    
    print("Obstacles detected:")
    for obs in obstacles:
        side = "right" if obs["angle"] > 0 else "left" if obs["angle"] < 0 else "center"
        print(f"  {obs['name']}: {obs['angle']:.1f}° ({side}), priority={obs['priority']}")
    
    # Calculate repulsion vectors
    print("\nCalculating repulsion vectors (pointing away from obstacles):")
    total_x = 0
    total_y = 0
    
    for obs in obstacles:
        # Create repulsion vector (opposite direction)
        repulsion_angle = obs["angle"] + 180
        if repulsion_angle > 180:
            repulsion_angle -= 360
        
        # Weight by priority and distance
        weight = obs["priority"] * (1 / max(obs["distance"], 0.1))
        
        # Convert to cartesian coordinates
        angle_rad = math.radians(repulsion_angle)
        vector_x = math.cos(angle_rad) * weight
        vector_y = math.sin(angle_rad) * weight
        
        total_x += vector_x
        total_y += vector_y
        
        print(f"  {obs['name']}: repulsion_angle={repulsion_angle:.1f}°, weight={weight:.1f}")
        print(f"    Vector: ({vector_x:.2f}, {vector_y:.2f})")
    
    # Calculate resultant direction
    print(f"\nResultant vector: ({total_x:.2f}, {total_y:.2f})")
    
    if total_x == 0 and total_y == 0:
        direction = DIRECTIONS['FORWARD']
        confidence = 0.5
    else:
        resultant_angle = math.degrees(math.atan2(total_y, total_x))
        confidence = min(1.0, math.sqrt(total_x**2 + total_y**2) / 10)
        
        # Map angle to navigation direction
        if -22.5 <= resultant_angle <= 22.5:
            direction = DIRECTIONS['FORWARD']
        elif 22.5 < resultant_angle <= 67.5:
            direction = DIRECTIONS['SLIGHT_RIGHT']
        elif 67.5 < resultant_angle <= 112.5:
            direction = DIRECTIONS['RIGHT']
        elif 112.5 < resultant_angle <= 157.5:
            direction = DIRECTIONS['SHARP_RIGHT']
        elif resultant_angle > 157.5 or resultant_angle < -157.5:
            direction = DIRECTIONS['STOP']
        elif -67.5 <= resultant_angle < -22.5:
            direction = DIRECTIONS['SLIGHT_LEFT']
        elif -112.5 <= resultant_angle < -67.5:
            direction = DIRECTIONS['LEFT']
        else:
            direction = DIRECTIONS['SHARP_LEFT']
    
    print(f"Resultant angle: {resultant_angle:.1f}°")
    print(f"Recommended direction: {direction}")
    print(f"Confidence: {confidence:.2f}")

def demo_simple_navigation():
    """Test the simple navigation system without YOLO"""
    print("\n" + "="*50)
    print("Simple Navigation System Demo")
    print("="*50)
    
    try:
        # Initialize navigation system
        nav = SimpleNavigation()
        print("✓ Navigation system initialized")
        print(f"  Image size: {nav.image_width}x{nav.image_height}")
        print(f"  Center point: ({nav.center_x}, {nav.center_y})")
        
        # Create demo image
        demo_image_path = create_demo_image()
        
        print("\nℹ️  Note: Since YOLO model download may take time,")
        print("   this demo shows the navigation logic without actual object detection.")
        print("   In real usage, YOLO would detect objects in the image automatically.")
        
        # For demonstration, we'll simulate what would happen if YOLO detected objects
        print("\n🔄 Simulating YOLO detection results...")
        
        # Simulate detection results
        simulated_detections = [
            {
                'bbox': [450, 200, 550, 400],
                'confidence': 0.85,
                'class_id': 0,
                'class_name': 'person'
            },
            {
                'bbox': [100, 250, 180, 350], 
                'confidence': 0.72,
                'class_id': 56,
                'class_name': 'chair'
            }
        ]
        
        # Analyze simulated detections
        spatial_objects = nav._analyze_spatial_objects(simulated_detections)
        
        print(f"✓ Analyzed {len(spatial_objects)} simulated objects")
        for obj in spatial_objects:
            print(f"  - {obj.class_name}: angle={obj.angle:.1f}°, distance={obj.distance:.2f}, priority={obj.priority}")
        
        # Generate navigation response
        result = nav._generate_navigation(spatial_objects)
        
        print(f"\n🧭 Navigation Result:")
        print(f"  Direction: {result['direction']}")
        print(f"  Instruction: {result['instruction']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        
        if result['warnings']:
            print(f"  Warnings: {result['warnings']}")
        
        print(f"\n📊 Safe Zones:")
        for zone in result['safe_zones']:
            status = "✓ Recommended" if zone['recommended'] else "✗ Not safe"
            print(f"  {zone['zone']}: {zone['safety_score']:.1f} {status}")
        
        # Clean up
        import os
        if os.path.exists(demo_image_path):
            os.unlink(demo_image_path)
            print(f"\n✓ Cleaned up demo image")
        
    except Exception as e:
        print(f"✗ Demo error: {e}")

def main():
    """Run all demos"""
    print("Simple Navigation System Demo")
    print("="*50)
    print("This demo shows how the navigation system works")
    print("using basic Python without external dependencies.")
    
    # Show available directions
    demo_navigation_directions()
    
    # Show vector calculations
    demo_vector_calculations()
    
    # Show direction calculation
    demo_direction_calculation()
    
    # Demo the navigation system
    demo_simple_navigation()
    
    print("\n" + "="*50)
    print("Demo completed!")
    print("\nTo use with real images:")
    print("  from simple_navigation import SimpleNavigation")
    print("  nav = SimpleNavigation()")
    print("  result = nav.analyze_image('your_image.jpeg')")
    print("\nTo start the API server:")
    print("  python3 simple_navigation_api.py")

if __name__ == "__main__":
    main() 