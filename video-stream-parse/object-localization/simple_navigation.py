#!/usr/bin/env python3
"""
Simple Navigation System for Visually Impaired Users
Takes JPEG image, runs YOLO detection, returns navigation direction
Uses only basic Python - no NumPy, OpenCV, dataclasses, enums, threading, etc.
"""

import math
import json
from PIL import Image
from ultralytics import YOLO

# Navigation directions (instead of enum)
DIRECTIONS = {
    'FORWARD': 'forward',
    'SLIGHT_LEFT': 'slight_left', 
    'LEFT': 'left',
    'SHARP_LEFT': 'sharp_left',
    'SLIGHT_RIGHT': 'slight_right',
    'RIGHT': 'right', 
    'SHARP_RIGHT': 'sharp_right',
    'STOP': 'stop'
}

# Object classes that are obstacles
OBSTACLE_CLASSES = {
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'suitcase', 'backpack', 'umbrella', 'handbag'
}

class DetectedObject:
    """Simple class to store object detection info"""
    def __init__(self, class_name, confidence, bbox, center, angle, distance, priority):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.center = center  # (x, y)
        self.angle = angle  # angle from center
        self.distance = distance  # estimated distance
        self.priority = priority  # navigation priority 1-10
        self.is_obstacle = class_name in OBSTACLE_CLASSES

class SimpleNavigation:
    """Simple navigation system"""
    
    def __init__(self, image_width=640, image_height=480):
        self.image_width = image_width
        self.image_height = image_height
        self.center_x = image_width // 2
        self.center_y = image_height // 2
        
        # Load YOLO model
        self.model = YOLO("yolo11n.pt")
    
    def analyze_image(self, image_path):
        """
        Main function: analyze image and return navigation guidance
        Args:
            image_path: path to JPEG image
        Returns:
            dict with navigation guidance
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Update image dimensions
            self.image_width, self.image_height = image.size
            self.center_x = self.image_width // 2
            self.center_y = self.image_height // 2
            
            # Run YOLO detection
            results = self.model(image, conf=0.5, verbose=False)
            
            # Extract detections
            detections = self._extract_detections(results)
            
            # Analyze for navigation
            spatial_objects = self._analyze_spatial_objects(detections)
            
            # Generate navigation instruction
            navigation_result = self._generate_navigation(spatial_objects)
            
            return navigation_result
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error analyzing image: {str(e)}',
                'direction': DIRECTIONS['STOP'],
                'instruction': 'Error - cannot analyze image',
                'confidence': 0.0
            }
    
    def _extract_detections(self, results):
        """Extract detection data from YOLO results"""
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                }
                detections.append(detection)
        
        return detections
    
    def _analyze_spatial_objects(self, detections):
        """Convert detections to spatial objects with navigation data"""
        spatial_objects = []
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Calculate center
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Calculate angle from image center (-90 to +90 degrees)
            rel_x = center_x - self.center_x
            angle = math.degrees(math.atan(rel_x / (self.image_width / 2))) if self.image_width > 0 else 0
            angle = max(-90, min(90, angle))
            
            # Estimate distance based on object size
            width = x2 - x1
            height = y2 - y1
            area = width * height
            image_area = self.image_width * self.image_height
            relative_area = area / image_area if image_area > 0 else 0
            
            # Distance estimation
            if relative_area > 0.3:
                distance = 0.2  # Very close
            elif relative_area > 0.15:
                distance = 0.4  # Close
            elif relative_area > 0.05:
                distance = 0.6  # Medium
            else:
                distance = 0.8  # Far
            
            # Adjust for object type
            if detection['class_name'] in ['person', 'car', 'truck', 'bus']:
                distance *= 0.8
            
            # Calculate priority
            priority = self._calculate_priority(detection, distance, angle)
            
            obj = DetectedObject(
                class_name=detection['class_name'],
                confidence=detection['confidence'],
                bbox=bbox,
                center=(center_x, center_y),
                angle=angle,
                distance=distance,
                priority=priority
            )
            
            spatial_objects.append(obj)
        
        return spatial_objects
    
    def _calculate_priority(self, detection, distance, angle):
        """Calculate navigation priority (1-10)"""
        priority = 1
        
        # Distance factor
        if distance < 0.3:
            priority += 4
        elif distance < 0.5:
            priority += 2
        
        # Angle factor (center path is high priority)
        if abs(angle) < 20:
            priority += 3
        elif abs(angle) < 45:
            priority += 1
        
        # Object type factor
        if detection['class_name'] in ['car', 'truck', 'bus']:
            priority += 3
        elif detection['class_name'] == 'person':
            priority += 2
        
        # Confidence factor
        if detection['confidence'] > 0.8:
            priority += 1
        
        return min(10, priority)
    
    def _generate_navigation(self, spatial_objects):
        """Generate navigation direction from spatial objects"""
        
        # If no objects, go forward
        if not spatial_objects:
            return {
                'status': 'success',
                'direction': DIRECTIONS['FORWARD'],
                'instruction': 'Continue straight ahead',
                'confidence': 0.9,
                'detected_objects': [],
                'warnings': []
            }
        
        # Calculate safe zones
        safe_zones = self._calculate_safe_zones(spatial_objects)
        
        # Find optimal direction using vector analysis
        direction, confidence = self._calculate_optimal_direction(spatial_objects)
        
        # Generate instruction
        instructions = {
            DIRECTIONS['FORWARD']: 'Continue straight ahead',
            DIRECTIONS['SLIGHT_LEFT']: 'Move slightly to the left',
            DIRECTIONS['LEFT']: 'Turn left',
            DIRECTIONS['SHARP_LEFT']: 'Turn sharply to the left',
            DIRECTIONS['SLIGHT_RIGHT']: 'Move slightly to the right',
            DIRECTIONS['RIGHT']: 'Turn right',
            DIRECTIONS['SHARP_RIGHT']: 'Turn sharply to the right',
            DIRECTIONS['STOP']: 'Stop - obstacles detected'
        }
        
        instruction = instructions.get(direction, 'Continue forward')
        
        # Generate warnings
        warnings = []
        for obj in spatial_objects:
            if obj.priority >= 8:
                if obj.class_name in ['car', 'truck', 'bus']:
                    warnings.append(f"Large vehicle detected: {obj.class_name}")
                elif obj.distance < 0.2:
                    warnings.append(f"Very close obstacle: {obj.class_name}")
                elif abs(obj.angle) < 10:
                    warnings.append(f"Obstacle directly ahead: {obj.class_name}")
        
        # Format detected objects for response
        detected_objects = []
        for obj in spatial_objects:
            detected_objects.append({
                'object': obj.class_name,
                'confidence': obj.confidence,
                'position': obj.center,
                'angle': obj.angle,
                'distance': obj.distance,
                'priority': obj.priority,
                'is_obstacle': obj.is_obstacle
            })
        
        return {
            'status': 'success',
            'direction': direction,
            'instruction': instruction,
            'confidence': confidence,
            'detected_objects': detected_objects,
            'safe_zones': safe_zones,
            'warnings': warnings
        }
    
    def _calculate_safe_zones(self, spatial_objects):
        """Calculate safety scores for different navigation zones"""
        zones = [
            (-90, -60, "far_left"),
            (-60, -30, "left"),
            (-30, -10, "slight_left"),
            (-10, 10, "center"),
            (10, 30, "slight_right"),
            (30, 60, "right"),
            (60, 90, "far_right")
        ]
        
        safe_zones = []
        
        for start_angle, end_angle, zone_name in zones:
            obstacles_in_zone = []
            
            for obj in spatial_objects:
                if obj.is_obstacle and start_angle <= obj.angle <= end_angle:
                    obstacles_in_zone.append(obj)
            
            # Calculate safety score
            safety_score = 10
            for obs in obstacles_in_zone:
                penalty = obs.priority * (1 / max(obs.distance, 0.1))
                safety_score -= penalty
                if obs.distance < 0.3:
                    safety_score -= 3
            
            safety_score = max(0, safety_score)
            
            safe_zones.append({
                'zone': zone_name,
                'angle_range': [start_angle, end_angle],
                'safety_score': safety_score,
                'obstacles': len(obstacles_in_zone),
                'recommended': safety_score >= 7
            })
        
        return safe_zones
    
    def _calculate_optimal_direction(self, spatial_objects):
        """Calculate optimal direction using vector analysis"""
        
        # Filter obstacles only
        obstacles = [obj for obj in spatial_objects if obj.is_obstacle]
        
        if not obstacles:
            return DIRECTIONS['FORWARD'], 0.9
        
        # Calculate repulsion vectors
        total_x = 0
        total_y = 0
        
        for obj in obstacles:
            # Create repulsion vector (pointing away from obstacle)
            repulsion_angle = obj.angle + 180  # Opposite direction
            if repulsion_angle > 180:
                repulsion_angle -= 360
            
            # Weight by priority and distance
            weight = obj.priority * (1 / max(obj.distance, 0.1))
            
            # Convert to cartesian coordinates
            angle_rad = math.radians(repulsion_angle)
            total_x += math.cos(angle_rad) * weight
            total_y += math.sin(angle_rad) * weight
        
        # Calculate resultant direction
        if total_x == 0 and total_y == 0:
            return DIRECTIONS['FORWARD'], 0.5
        
        resultant_angle = math.degrees(math.atan2(total_y, total_x))
        
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
        
        # Calculate confidence
        confidence = min(1.0, math.sqrt(total_x**2 + total_y**2) / 10)
        
        return direction, confidence

def test_simple_navigation():
    """Test the simple navigation system"""
    print("Testing Simple Navigation System")
    print("=" * 40)
    
    # Initialize navigation
    nav = SimpleNavigation()
    
    # Test with sample image (you need to provide an actual image)
    # For now, we'll simulate the process
    print("✓ Navigation system initialized")
    print("ℹ️  To test with real image, call: nav.analyze_image('your_image.jpg')")
    
    # Example output format
    sample_result = {
        'status': 'success',
        'direction': 'slight_left',
        'instruction': 'Move slightly to the left',
        'confidence': 0.75,
        'detected_objects': [
            {
                'object': 'person',
                'confidence': 0.85,
                'position': (400, 300),
                'angle': 25.0,
                'distance': 0.4,
                'priority': 7,
                'is_obstacle': True
            }
        ],
        'safe_zones': [
            {'zone': 'left', 'safety_score': 8.5, 'recommended': True},
            {'zone': 'center', 'safety_score': 3.2, 'recommended': False}
        ],
        'warnings': []
    }
    
    print("\nExample output format:")
    print(json.dumps(sample_result, indent=2))
    
    return nav

if __name__ == "__main__":
    nav_system = test_simple_navigation() 