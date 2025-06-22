# Simple Navigation System for Visually Impaired Users

A simplified navigation system that takes JPEG images, processes them through YOLO object detection, and provides navigation directions using only basic Python libraries.

## Overview

This system provides navigation assistance by:
1. Taking a JPEG image as input
2. Running YOLO object detection to find obstacles
3. Calculating spatial relationships between objects
4. Using vector mathematics to determine safe navigation directions
5. Returning human-readable navigation instructions

## Key Features

✅ **No External Dependencies**: Uses only basic Python libraries (math, json)  
✅ **No DataClasses**: Uses simple Python classes  
✅ **No Enums**: Uses dictionary constants  
✅ **No NumPy/OpenCV**: Pure Python implementation  
✅ **No Threading**: Synchronous processing  
✅ **Vector Mathematics**: Proper obstacle avoidance using repulsion vectors  
✅ **Safe Zone Analysis**: 7-zone safety scoring system  

## Files

### Core System
- **`minimal_navigation.py`** - Core navigation logic (no external dependencies)
- **`simple_navigation.py`** - Full navigation with YOLO integration  
- **`simple_navigation_api.py`** - HTTP API server for image analysis

### Demos and Tests
- **`demo_simple_navigation.py`** - Demo without YOLO (shows logic)
- **`test_simple_navigation.py`** - Full test with API integration

## Quick Start

### 1. Minimal Demo (No Dependencies)

```bash
python3 minimal_navigation.py
```

This shows the core navigation logic with sample data.

### 2. Basic Usage

```python
from minimal_navigation import MinimalNavigation

# Initialize navigation system
nav = MinimalNavigation()

# Sample YOLO detection data
detections = [
    {
        'bbox': [450, 200, 550, 400],  # [x1, y1, x2, y2]
        'confidence': 0.85,
        'class_name': 'person'
    }
]

# Get navigation guidance
result = nav.analyze_detection_data(detections)

print(f"Direction: {result['direction']}")
print(f"Instruction: {result['instruction']}")
print(f"Confidence: {result['confidence']}")
```

### 3. API Server

```bash
python3 simple_navigation_api.py
```

Then send POST requests to `http://localhost:5000/analyze` with base64-encoded images.

## Input Format

The system expects detection data in this format:

```python
detection_list = [
    {
        'bbox': [x1, y1, x2, y2],  # Bounding box coordinates
        'confidence': 0.85,        # Detection confidence (0.0-1.0)
        'class_name': 'person'     # Object class name
    }
]
```

## Output Format

```python
{
    'status': 'success',
    'direction': 'slight_left',                    # Navigation direction
    'instruction': 'Move slightly to the left',    # Human-readable instruction
    'confidence': 0.75,                           # Confidence in recommendation
    'detected_objects': [
        {
            'object': 'person',
            'confidence': 0.85,
            'position': (500, 300),
            'angle': 29.4,              # Angle from center (-90 to +90)
            'distance': 0.48,           # Estimated distance (0.0-1.0)
            'priority': 7,              # Navigation priority (1-10)
            'is_obstacle': True
        }
    ],
    'safe_zones': [
        {
            'zone': 'far_left',
            'safety_score': 10.0,       # Safety score (0-10)
            'obstacles': 0,
            'recommended': True
        }
    ],
    'warnings': ['Large vehicle detected: car']
}
```

## Navigation Directions

The system provides these navigation directions:

- **`forward`** - Continue straight ahead
- **`slight_left`** - Move slightly to the left  
- **`left`** - Turn left
- **`sharp_left`** - Turn sharply to the left
- **`slight_right`** - Move slightly to the right
- **`right`** - Turn right
- **`sharp_right`** - Turn sharply to the right
- **`stop`** - Stop - obstacles detected

## Vector Mathematics

The system uses vector mathematics for navigation:

1. **Object Detection**: YOLO provides bounding boxes
2. **Spatial Analysis**: Calculate object centers and angles
3. **Repulsion Vectors**: Create vectors pointing away from obstacles  
4. **Vector Summation**: Sum all repulsion vectors weighted by priority
5. **Direction Mapping**: Convert resultant vector to navigation direction

### Example Vector Calculation

```
Obstacle at 35° (right side):
- Repulsion angle: 35° + 180° = 215° → -145°
- Weight: priority × (1/distance) = 8 × (1/0.4) = 20.0
- Vector: (-16.38, -11.47)

Resultant direction points left, so recommend "turn left"
```

## Safe Zones

The system divides the field of view into 7 zones:

- **far_left**: -90° to -60°
- **left**: -60° to -30°  
- **slight_left**: -30° to -10°
- **center**: -10° to +10°
- **slight_right**: +10° to +30°
- **right**: +30° to +60°
- **far_right**: +60° to +90°

Each zone gets a safety score (0-10) based on obstacles present.

## Priority System

Objects get navigation priority (1-10) based on:

- **Distance**: Closer objects get higher priority
- **Angle**: Objects in center path get higher priority  
- **Object Type**: Vehicles and people get higher priority
- **Confidence**: Higher confidence detections get higher priority

## API Usage

### Start Server
```bash
python3 simple_navigation_api.py
```

### Analyze Image
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<base64_encoded_image>"}'
```

### Test Endpoints
```bash
# API status
curl http://localhost:5000/status

# Sample navigation data  
curl http://localhost:5000/test
```

## Example Output

```
🧭 Navigation Result:
  Direction: slight_left
  Instruction: Move slightly to the left
  Confidence: 0.75

📍 Detected Objects:
  - person: 29.4° (right), distance=0.48, priority=7 🚫 OBSTACLE
  - chair: -29.4° (left), distance=0.80, priority=2 🚫 OBSTACLE

📊 Safe Zones:
  ✓ far_left: safety=10.0 (RECOMMENDED)
  ✓ slight_left: safety=7.5 (RECOMMENDED)
  ✗ center: safety=0.0 (avoid)

⚠️ Warnings:
  - Obstacle directly ahead: person
```

## Integration with YOLO

To use with actual YOLO detection:

```python
from ultralytics import YOLO
from simple_navigation import SimpleNavigation

# Load YOLO model
model = YOLO("yolo11n.pt")
nav = SimpleNavigation()

# Process image
results = model("image.jpg", conf=0.5)
detections = []

for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    confidence = float(box.conf[0].cpu().numpy())
    class_name = model.names[int(box.cls[0].cpu().numpy())]
    
    detections.append({
        'bbox': [float(x1), float(y1), float(x2), float(y2)],
        'confidence': confidence,
        'class_name': class_name
    })

# Get navigation guidance
result = nav.analyze_image("image.jpg")  # or analyze_detection_data(detections)
```

## Troubleshooting

### Common Issues

1. **YOLO Model Download**: First run may take time to download model
2. **Image Format**: Ensure images are JPEG format
3. **Detection Format**: Check bbox coordinates are [x1,y1,x2,y2]

### Dependencies for Full System

```bash
pip install ultralytics  # For YOLO
pip install Pillow      # For image processing
```

### For Minimal System
No additional dependencies required - uses only Python standard library!

## License

MIT License - See LICENSE file for details. 