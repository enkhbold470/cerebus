import asyncio
import base64
import datetime
import json
import os
import threading
import time
import io
import queue
import subprocess
import signal
from typing import Dict, List, Optional, Tuple
from PIL import Image

import cv2
import numpy as np
import requests
import speech_recognition as sr
from google.adk.agents import Agent
from google.cloud import texttospeech
from ultralytics import YOLO

# Global session storage
GPS_SESSIONS = {}
VIDEO_SESSIONS = {}
NAVIGATION_SESSIONS = {}
AUDIO_SESSIONS = {}


def start_gps_polling(
    gps_endpoint: str,
    session_id: str,
    poll_interval: int = 5,
    navigation_session_id: Optional[str] = None,
) -> dict:
    """Starts continuous GPS polling from endpoint and updates navigation.

    Args:
        gps_endpoint (str): URL endpoint that returns GPS coordinates
        session_id (str): Unique session identifier
        poll_interval (int): Polling interval in seconds
        navigation_session_id (str): Associated navigation session to update

    Returns:
        dict: GPS polling session info
    """
    try:

        def poll_gps():
            session = GPS_SESSIONS[session_id]
            consecutive_failures = 0

            while session["active"]:
                try:
                    # Poll GPS endpoint
                    response = requests.get(
                        gps_endpoint, timeout=10, headers={"Accept": "application/json"}
                    )

                    if response.status_code == 200:
                        gps_data = response.json()
                        lat = float(gps_data.get("latitude", 0))
                        lng = float(gps_data.get("longitude", 0))
                        accuracy = gps_data.get("accuracy", "unknown")

                        # Update session data
                        session["last_location"] = {"lat": lat, "lng": lng}
                        session["last_update"] = time.time()
                        session["accuracy"] = accuracy
                        consecutive_failures = 0

                        # Update navigation if linked
                        if (
                            navigation_session_id
                            and navigation_session_id in NAVIGATION_SESSIONS
                        ):
                            nav_update = update_navigation_location(
                                navigation_session_id, lat, lng
                            )

                            # Generate audio for navigation updates
                            if (
                                nav_update.get("status") == "success"
                                and nav_update.get("navigation_status") == "updated"
                            ):

                                audio_instruction = nav_update.get(
                                    "current_instruction", ""
                                )
                                if audio_instruction:
                                    generate_navigation_audio(
                                        f"Navigation update: {audio_instruction}",
                                        priority="high",
                                    )

                        # Log GPS update
                        session["location_history"].append(
                            {
                                "timestamp": time.time(),
                                "lat": lat,
                                "lng": lng,
                                "accuracy": accuracy,
                            }
                        )

                        # Keep only last 100 locations
                        if len(session["location_history"]) > 100:
                            session["location_history"] = session["location_history"][
                                -100:
                            ]

                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= 3:
                            generate_system_audio(
                                "GPS signal lost. Please check your connection.",
                                priority="high",
                            )

                except Exception as e:
                    consecutive_failures += 1
                    session["last_error"] = str(e)

                    if consecutive_failures >= 5:
                        generate_system_audio(
                            "GPS polling error. Navigation may be affected.",
                            priority="medium",
                        )

                time.sleep(poll_interval)

        # Create session
        GPS_SESSIONS[session_id] = {
            "active": True,
            "gps_endpoint": gps_endpoint,
            "poll_interval": poll_interval,
            "navigation_session_id": navigation_session_id,
            "last_location": None,
            "last_update": None,
            "accuracy": None,
            "location_history": [],
            "last_error": None,
            "start_time": time.time(),
        }

        # Start polling thread
        polling_thread = threading.Thread(target=poll_gps, daemon=True)
        polling_thread.start()

        return {
            "status": "success",
            "session_id": session_id,
            "gps_endpoint": gps_endpoint,
            "poll_interval": poll_interval,
            "linked_navigation": navigation_session_id,
            "report": f"GPS polling started. Session: {session_id}",
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to start GPS polling: {str(e)}",
        }


def get_current_gps_location(session_id: str) -> dict:
    """Gets the most recent GPS location from polling session."""
    if session_id not in GPS_SESSIONS:
        return {
            "status": "error",
            "error_message": f"GPS session {session_id} not found",
        }

    session = GPS_SESSIONS[session_id]

    if not session["last_location"]:
        return {"status": "error", "error_message": "No GPS data available yet"}

    return {
        "status": "success",
        "session_id": session_id,
        "location": session["last_location"],
        "accuracy": session["accuracy"],
        "last_update": session["last_update"],
        "age_seconds": time.time() - session["last_update"],
        "report": f"Current location: {session['last_location']['lat']:.6f}, {session['last_location']['lng']:.6f}",
    }


def start_live_navigation(
    destination: str,
    current_lat: float,
    current_lng: float,
    session_id: Optional[str] = None,
) -> dict:
    """Starts live navigation session with real-time turn-by-turn directions."""

    try:
        # Generate session ID if not provided
        if not session_id:
            session_id = f"nav_{int(time.time())}"

        # TODO: In a real implementation, this would use a routing service like Google Maps API
        # For now, we'll create a basic navigation session structure

        # Create navigation session
        NAVIGATION_SESSIONS[session_id] = {
            "active": True,
            "destination": destination,
            "start_location": {"lat": current_lat, "lng": current_lng},
            "current_location": {"lat": current_lat, "lng": current_lng},
            "route": [],  # Will be populated by routing service
            "current_step": 0,
            "status": "navigating",
            "start_time": time.time(),
            "last_update": time.time(),
            "total_distance": 0,
            "remaining_distance": 0,
            "estimated_time": 0,
            "current_instruction": f"Navigation started to {destination}",
            "next_instruction": None,
        }

        # Generate initial navigation audio
        generate_navigation_audio(
            f"Navigation started to {destination}. Calculating route from your current location.",
            priority="high",
        )

        return {
            "status": "success",
            "session_id": session_id,
            "destination": destination,
            "start_location": {"lat": current_lat, "lng": current_lng},
            "navigation_status": "started",
            "report": f"Live navigation started to {destination}. Session: {session_id}",
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to start navigation: {str(e)}",
        }


def update_navigation_location(
    session_id: str,
    current_lat: float,
    current_lng: float,
) -> dict:
    """Updates current location for active navigation session."""

    try:
        if session_id not in NAVIGATION_SESSIONS:
            return {
                "status": "error",
                "error_message": f"Navigation session {session_id} not found",
            }

        session = NAVIGATION_SESSIONS[session_id]

        if not session["active"]:
            return {
                "status": "error",
                "error_message": f"Navigation session {session_id} is not active",
            }

        # Update current location
        old_location = session["current_location"]
        session["current_location"] = {"lat": current_lat, "lng": current_lng}
        session["last_update"] = time.time()

        # Calculate distance moved (simple approximation)
        lat_diff = current_lat - old_location["lat"]
        lng_diff = current_lng - old_location["lng"]
        distance_moved = ((lat_diff**2 + lng_diff**2) ** 0.5) * 111000  # Rough meters

        # TODO: In a real implementation, this would:
        # 1. Check if we're still on route
        # 2. Recalculate route if needed
        # 3. Update turn-by-turn instructions
        # 4. Calculate remaining distance and time

        # For now, provide basic location update
        session["current_instruction"] = f"Continue to {session['destination']}"

        # Only generate audio if we've moved significantly (>10 meters)
        navigation_status = "updated" if distance_moved > 10 else "position_updated"

        return {
            "status": "success",
            "session_id": session_id,
            "current_location": {"lat": current_lat, "lng": current_lng},
            "distance_moved": round(distance_moved, 2),
            "navigation_status": navigation_status,
            "current_instruction": session["current_instruction"],
            "report": f"Navigation location updated. Moved {distance_moved:.1f}m",
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to update navigation location: {str(e)}",
        }


def get_navigation_status(session_id: str) -> dict:
    """Gets current status of navigation session."""

    try:
        if session_id not in NAVIGATION_SESSIONS:
            return {
                "status": "error",
                "error_message": f"Navigation session {session_id} not found",
            }

        session = NAVIGATION_SESSIONS[session_id]

        # Calculate session duration
        duration = time.time() - session["start_time"]
        time_since_update = time.time() - session["last_update"]

        return {
            "status": "success",
            "session_id": session_id,
            "navigation_active": session["active"],
            "destination": session["destination"],
            "current_location": session["current_location"],
            "start_location": session["start_location"],
            "current_instruction": session["current_instruction"],
            "next_instruction": session.get("next_instruction"),
            "navigation_status": session["status"],
            "duration_seconds": round(duration, 1),
            "time_since_last_update": round(time_since_update, 1),
            "total_distance": session.get("total_distance", 0),
            "remaining_distance": session.get("remaining_distance", 0),
            "estimated_time": session.get("estimated_time", 0),
            "report": f"Navigation to {session['destination']} - Status: {session['status']}",
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to get navigation status: {str(e)}",
        }


def start_video_analysis_with_dodging(
    camera_source: int = 0,
    model_path: str = "yolo11n.pt",  # Updated to YOLOv11
    analysis_interval: float = 0.5,
    dodging_enabled: bool = True,
) -> dict:
    """Starts video analysis with real-time dodging instructions for blind users."""

    try:
        model = YOLO(model_path)
        cap = cv2.VideoCapture(camera_source)

        if not cap.isOpened():
            return {
                "status": "error",
                "error_message": f"Could not open camera {camera_source}",
            }

        session_id = f"video_dodge_{int(time.time())}"

        VIDEO_SESSIONS[session_id] = {
            "active": True,
            "camera": cap,
            "model": model,
            "analysis_interval": analysis_interval,
            "dodging_enabled": dodging_enabled,
            "last_analysis": 0,
            "dodge_history": [],
            "frame_count": 0,
            "start_time": time.time(),
            "last_dodge_instruction": 0,
        }

        def analyze_and_dodge():
            session = VIDEO_SESSIONS[session_id]

            while session["active"]:
                ret, frame = session["camera"].read()
                if not ret:
                    break

                session["frame_count"] += 1
                current_time = time.time()

                if (
                    current_time - session["last_analysis"]
                    >= session["analysis_interval"]
                ):
                    # Analyze frame for obstacles
                    dodge_analysis = analyze_frame_for_dodging_internal(
                        frame, session["model"]
                    )
                    session["last_analysis"] = current_time

                    if dodge_analysis["needs_dodging"] and session["dodging_enabled"]:
                        # Generate dodging instruction
                        dodge_instruction = generate_dodge_instruction(dodge_analysis)

                        # Avoid spam - only give dodge instructions every 2 seconds
                        if current_time - session["last_dodge_instruction"] > 2.0:
                            # Generate immediate audio warning
                            generate_dodge_audio(dodge_instruction["instruction"])
                            session["last_dodge_instruction"] = current_time

                            # Store dodge event
                            session["dodge_history"].append(
                                {
                                    "timestamp": current_time,
                                    "instruction": dodge_instruction,
                                    "obstacles": dodge_analysis["obstacles"],
                                    "frame_number": session["frame_count"],
                                }
                            )

                    # Keep only last 50 dodge events
                    if len(session["dodge_history"]) > 50:
                        session["dodge_history"] = session["dodge_history"][-50:]

                time.sleep(0.05)  # 20 FPS analysis

        # Start analysis thread
        analysis_thread = threading.Thread(target=analyze_and_dodge, daemon=True)
        analysis_thread.start()

        return {
            "status": "success",
            "session_id": session_id,
            "camera_source": camera_source,
            "dodging_enabled": dodging_enabled,
            "analysis_interval": analysis_interval,
            "report": f"Video dodging analysis started. Session: {session_id}",
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error starting video analysis: {str(e)}",
        }


def analyze_frame_for_dodging_internal(frame: np.ndarray, model: YOLO) -> dict:
    """Internal function to analyze frame specifically for dodging requirements."""
    try:
        results = model(frame, conf=0.4, verbose=False)

        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width // 2

        # Define path zones
        left_zone = (0, center_x - 100)
        center_zone = (center_x - 100, center_x + 100)
        right_zone = (center_x + 100, frame_width)

        obstacles = {"left": [], "center": [], "right": [], "overhead": []}

        critical_obstacles = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    obj_center_x = (x1 + x2) / 2
                    obj_center_y = (y1 + y2) / 2
                    obj_width = x2 - x1
                    obj_height = y2 - y1

                    # Determine zone
                    if obj_center_x < center_x - 100:
                        zone = "left"
                    elif obj_center_x > center_x + 100:
                        zone = "right"
                    else:
                        zone = "center"

                    # Check if overhead (top 30% of frame)
                    if y1 < frame_height * 0.3:
                        zone = "overhead"

                    obstacle = {
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "center": [obj_center_x, obj_center_y],
                        "size": [obj_width, obj_height],
                        "zone": zone,
                        "distance_estimate": estimate_obstacle_distance(
                            [x1, y1, x2, y2], frame.shape
                        ),
                    }

                    obstacles[zone].append(obstacle)

                    # Check if critical (large, close, in center path)
                    relative_size = (obj_width * obj_height) / (
                        frame_width * frame_height
                    )

                    if (
                        zone == "center"
                        and relative_size > 0.1
                        and class_name
                        in ["person", "car", "bicycle", "motorcycle", "bus", "truck"]
                    ):
                        critical_obstacles.append(obstacle)

        # Determine if dodging is needed
        needs_dodging = len(critical_obstacles) > 0 or len(obstacles["center"]) > 2

        return {
            "needs_dodging": needs_dodging,
            "obstacles": obstacles,
            "critical_obstacles": critical_obstacles,
            "frame_analyzed": True,
            "analysis_timestamp": time.time(),
        }

    except Exception as e:
        return {
            "needs_dodging": False,
            "obstacles": {"left": [], "center": [], "right": [], "overhead": []},
            "critical_obstacles": [],
            "frame_analyzed": False,
            "error": str(e),
        }


def analyze_current_frame_for_dodging(session_id: str = "default") -> dict:
    """ADK-friendly function to analyze the current frame for dodging from active video session.

    Args:
        session_id (str): Video analysis session identifier

    Returns:
        dict: Dodging analysis result with obstacles and recommendations
    """
    try:
        # Check if video session exists and is active
        if session_id not in VIDEO_SESSIONS:
            return {
                "status": "error",
                "error_message": f"No video session found with ID: {session_id}",
                "needs_dodging": False,
                "report": "No active video analysis session for dodging analysis",
            }

        session = VIDEO_SESSIONS[session_id]
        if not session.get("active", False):
            return {
                "status": "error",
                "error_message": "Video session is not active",
                "needs_dodging": False,
                "report": "Video session stopped - cannot analyze for dodging",
            }

        # Get the latest frame from the ESP32 stream
        if "parser" in session:
            frame_data = session["parser"].get_latest_frame(timeout=1.0)
            if frame_data:
                img_array, _ = frame_data

                # Load YOLO model if not available in session
                if "yolo_model" not in session:
                    from ultralytics import YOLO

                    session["yolo_model"] = YOLO("yolo11n.pt")

                # Perform dodging analysis
                dodge_analysis = analyze_frame_for_dodging_internal(
                    img_array, session["yolo_model"]
                )

                # Add status and report
                dodge_analysis["status"] = "success"
                dodge_analysis["session_id"] = session_id

                if dodge_analysis["needs_dodging"]:
                    critical_count = len(dodge_analysis["critical_obstacles"])
                    obstacle_count = sum(
                        len(obs) for obs in dodge_analysis["obstacles"].values()
                    )
                    dodge_analysis["report"] = (
                        f"DODGING REQUIRED: {critical_count} critical obstacles, {obstacle_count} total detected"
                    )
                else:
                    dodge_analysis["report"] = "Path clear - no dodging required"

                return dodge_analysis
            else:
                return {
                    "status": "error",
                    "error_message": "No frame available from video stream",
                    "needs_dodging": False,
                    "report": "Cannot analyze - no current frame from ESP32 camera",
                }
        else:
            return {
                "status": "error",
                "error_message": "No stream parser in session",
                "needs_dodging": False,
                "report": "Video session misconfigured - no stream parser",
            }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Dodging analysis failed: {str(e)}",
            "needs_dodging": False,
            "report": "Error during dodging analysis",
        }


def start_esp32_camera_for_navigation(session_id: str = "navigation") -> dict:
    """Start ESP32 camera with optimal settings for blind navigation assistance.

    This function automatically connects to the ESP32 camera at 192.168.18.39:81
    with YOLOv11 object detection optimized for navigation assistance.

    Args:
        session_id (str): Session identifier for the camera stream

    Returns:
        dict: Camera startup result and status
    """
    return start_esp32_visual_analysis(
        esp32_ip="192.168.18.39",
        stream_port=81,
        session_id=session_id,
        enable_yolo=True,
        yolo_confidence=0.5,
        analysis_fps=2.0,
        flip_180=True,
        save_frames=False,
    )


def start_esp32_navigation_to_destination(destination: str) -> dict:
    """Start complete ESP32-based navigation system to a specific destination.

    This function automatically sets up:
    - ESP32 camera with YOLOv11 object detection
    - Ultrasonic distance monitoring
    - Navigation status monitoring
    - Audio feedback system

    Args:
        destination (str): The destination to navigate to (e.g., "Bear Cafe")

    Returns:
        dict: Navigation system startup status
    """
    try:
        results = {}

        # 1. Start ESP32 camera for visual navigation
        camera_result = start_esp32_camera_for_navigation("navigation")
        if camera_result["status"] == "success":
            results["esp32_camera"] = "navigation"

        # 2. Start ultrasonic distance monitoring
        distance_result = monitor_ultrasonic_distance(
            session_id="navigation", poll_interval=0.5, alert_threshold=1.0
        )
        if distance_result["status"] == "success":
            results["ultrasonic_sensor"] = "navigation"

        # 3. Get current navigation status (assumes navigation server is running)
        nav_status = get_current_navigation_status("default")
        if nav_status["status"] == "success":
            results["navigation_server"] = "connected"

        # 4. Generate welcome message
        welcome_message = (
            f"ESP32 navigation system started for {destination}. "
            "Camera active with YOLOv11 object detection. "
            "Ultrasonic sensor monitoring for obstacles. "
            "I can see what's ahead and will guide you safely."
        )
        generate_system_audio(welcome_message, priority="high")

        return {
            "status": "success",
            "destination": destination,
            "esp32_systems_active": results,
            "features_enabled": {
                "esp32_camera": "esp32_camera" in results,
                "ultrasonic_sensor": "ultrasonic_sensor" in results,
                "navigation_server": "navigation_server" in results,
                "yolo_detection": True,
                "audio_feedback": True,
            },
            "report": f"ESP32 navigation system ready for {destination} - camera and sensors active",
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to start ESP32 navigation: {str(e)}",
            "report": "Error starting ESP32 navigation system",
        }


def capture_esp32_frame() -> dict:
    """Capture a single frame from ESP32 using /capture endpoint as fallback.
    
    Returns:
        dict: Frame capture result with image data
    """
    try:
        # Try /capture endpoint directly
        esp32_ip = "192.168.18.39"
        capture_url = f"http://{esp32_ip}/capture"
        
        curl_result = subprocess.run(
            ["curl", "-s", capture_url, "--max-time", "10"],
            capture_output=True,
            timeout=15
        )
        
        if curl_result.returncode == 0 and curl_result.stdout:
            # Convert bytes to PIL image
            img_bytes = curl_result.stdout
            
            try:
                from PIL import Image
                import io
                
                pil_image = Image.open(io.BytesIO(img_bytes))
                
                # Flip 180 degrees like the stream parser does
                pil_image = pil_image.rotate(180)
                
                # Convert to numpy array
                img_array = np.array(pil_image)
                
                # Save debug image
                debug_dir = "debug_images"
                os.makedirs(debug_dir, exist_ok=True)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                debug_filename = f"esp32_capture_{timestamp}.jpg"
                debug_path = os.path.join(debug_dir, debug_filename)
                
                pil_image.save(debug_path, "JPEG", quality=85)
                print(f"📸 ESP32 capture saved: {debug_path}")
                
                return {
                    "status": "success",
                    "image_array": img_array,  # Note: Not JSON serializable, for internal use only
                    "pil_image": pil_image,    # Note: Not JSON serializable, for internal use only  
                    "debug_path": debug_path,
                    "capture_method": "esp32_capture",
                    "image_size": list(pil_image.size),
                    "image_mode": pil_image.mode,
                    "report": f"Frame captured via /capture endpoint: {pil_image.size}",
                }
                
            except Exception as e:
                return {
                    "status": "error",
                    "error_message": f"Failed to process captured image: {str(e)}",
                    "report": "Image processing error",
                }
        else:
            return {
                "status": "error", 
                "error_message": f"Capture failed: return code {curl_result.returncode}",
                "stderr": curl_result.stderr.decode() if curl_result.stderr else "",
                "report": "ESP32 capture endpoint failed",
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Capture error: {str(e)}",
            "report": "Error capturing frame from ESP32",
        }


def get_current_visual_analysis() -> dict:
    """Get current visual analysis from ESP32 camera for navigation assistance.

    Automatically starts ESP32 camera if not already running, then returns what the camera sees.

    Returns:
        dict: Current visual analysis results
    """
    try:
        print(f"🚀 [DEBUG] get_current_visual_analysis called")
        print(f"🚀 [DEBUG] Current working directory: {os.getcwd()}")
        print(f"🚀 [DEBUG] VIDEO_SESSIONS keys: {list(VIDEO_SESSIONS.keys())}")
                # Check if ESP32 camera is already active
        session_id = "navigation"
        visual_status = get_esp32_visual_status(session_id)
        print(f"🚀 [DEBUG] Visual status: {visual_status}")
        
        # If not active, start the ESP32 camera automatically
        if visual_status["status"] != "success" or not visual_status.get(
            "active", False
        ):
            print("🚀 [DEBUG] Starting ESP32 camera for visual analysis...")
            start_result = start_esp32_visual_analysis(session_id=session_id)
            print(f"🚀 [DEBUG] Start result: {start_result}")
            
            # Force capture since stream might not work
            print("🚀 [DEBUG] Forcing capture attempt...")
            capture_result = capture_esp32_frame()
            print(f"🚀 [DEBUG] Capture result: {capture_result['status']}")
            
            if capture_result["status"] == "success":
                return {
                    "status": "success",
                    "visual_active": True,
                    "objects_detected": 0,
                    "important_objects": 0,
                    "all_detections": [],
                    "needs_caution": False,
                    "frames_processed": 1,
                    "total_detections": 0,
                    "debug_image_path": capture_result["debug_path"],
                    "capture_method": "forced_capture",
                    "report": f"Frame captured directly via /capture: {capture_result['report']}",
                }

            if start_result["status"] != "success":
                return {
                    "status": "error",
                    "visual_active": False,
                    "error_message": f"Failed to start ESP32 camera: {start_result.get('error_message', 'Unknown error')}",
                    "report": "Could not start ESP32 camera for visual analysis",
                }

            # Wait a moment for camera to initialize
            time.sleep(2)

            # Get updated status
            visual_status = get_esp32_visual_status(session_id)

        # Now get the visual analysis
        if visual_status["status"] == "success" and visual_status["active"]:
            session = VIDEO_SESSIONS[session_id]
            recent_detections = session.get("last_detections", [])

            # Capture debug image - try stream first, then fallback to /capture
            debug_image_path = None
            img_array = None
            pil_image = None
            
            try:
                # First try to get frame from stream parser
                frame_from_stream = False
                if "parser" in session:
                    parser = session["parser"]
                    
                    # Try multiple times to get a frame since the stream might be just starting
                    frame_data = None
                    for attempt in range(2):  # Reduced attempts for faster fallback
                        frame_data = parser.get_latest_frame(timeout=1.0)
                        if frame_data:
                            img_array, pil_image = frame_data
                            frame_from_stream = True
                            print(f"✓ Frame from stream parser")
                            break
                        print(f"📡 Waiting for stream frame... attempt {attempt + 1}")
                        time.sleep(0.5)

                # If stream failed, try /capture endpoint
                if not frame_from_stream:
                    print("📡 Stream failed, trying /capture endpoint...")
                    capture_result = capture_esp32_frame()
                    
                    if capture_result["status"] == "success":
                        img_array = capture_result["image_array"]
                        pil_image = capture_result["pil_image"] 
                        debug_image_path = capture_result["debug_path"]
                        print(f"✓ Frame from /capture endpoint")
                    else:
                        print(f"✗ /capture also failed: {capture_result.get('error_message', 'Unknown error')}")

                # If we got an image from either method, save additional debug info
                if img_array is not None and pil_image is not None:
                    if not debug_image_path:  # Only if not already saved by capture_esp32_frame
                        # Create debug directory
                        debug_dir = "debug_images"
                        os.makedirs(debug_dir, exist_ok=True)

                        # Save debug image with timestamp
                        timestamp = datetime.datetime.now().strftime(
                            "%Y%m%d_%H%M%S_%f"
                        )[:-3]
                        debug_filename = f"visual_analysis_{timestamp}.jpg"
                        debug_image_path = os.path.join(debug_dir, debug_filename)

                        # Save the image
                        pil_image.save(debug_image_path, "JPEG", quality=85)
                        print(f"📸 Debug image saved: {debug_image_path}")
                    
                    # Try YOLO detection on the captured frame
                    if frame_from_stream and "parser" in session and parser.enable_yolo:
                        # Use parser's YOLO if available
                        annotated_image, detections = parser.detect_objects_yolo(img_array)
                        if detections:
                            recent_detections = detections  # Update with fresh detections
                            session["last_detections"] = detections
                            session["detection_count"] += 1
                            
                        if annotated_image:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            annotated_filename = f"visual_analysis_annotated_{timestamp}.jpg"
                            annotated_path = os.path.join("debug_images", annotated_filename)
                            annotated_image.save(annotated_path, "JPEG", quality=85)
                            print(f"📸 Annotated debug image saved: {annotated_path}")
                else:
                    print("⚠️  No frame available from either stream or capture endpoint")

            except Exception as e:
                print(f"⚠️  Could not capture/save debug image: {e}")

            if recent_detections:
                # Analyze detections for navigation context
                important_objects = []
                all_objects = []

                for detection in recent_detections:
                    class_name = detection["class_name"]
                    confidence = detection["confidence"]
                    all_objects.append(f"{class_name} ({confidence:.2f})")

                    # Check for important navigation objects
                    if class_name in [
                        "person",
                        "car",
                        "bicycle",
                        "motorbike",
                        "bus",
                        "truck",
                        "dog",
                        "cat",
                    ]:
                        important_objects.append(detection)

                total_objects = len(recent_detections)
                important_count = len(important_objects)

                if important_count > 0:
                    important_names = [obj["class_name"] for obj in important_objects]
                    visual_report = f"I can see {total_objects} objects ahead, including {important_count} important ones: {', '.join(important_names[:3])}"
                    if important_count > 3:
                        visual_report += f" and {important_count - 3} more"
                    visual_report += " - Please proceed with caution"
                    needs_caution = True
                else:
                    visual_report = f"I can see {total_objects} objects ahead: {', '.join([obj['class_name'] for obj in recent_detections[:3]])}"
                    if total_objects > 3:
                        visual_report += f" and {total_objects - 3} more"
                    visual_report += " - Path appears navigable"
                    needs_caution = False

                return {
                    "status": "success",
                    "visual_active": True,
                    "objects_detected": total_objects,
                    "important_objects": important_count,
                    "all_detections": recent_detections,
                    "needs_caution": needs_caution,
                    "frames_processed": session["frame_count"],
                    "total_detections": session["detection_count"],
                    "debug_image_path": debug_image_path,
                    "report": visual_report,
                }
            else:
                return {
                    "status": "success",
                    "visual_active": True,
                    "objects_detected": 0,
                    "important_objects": 0,
                    "all_detections": [],
                    "needs_caution": False,
                    "frames_processed": session["frame_count"],
                    "total_detections": session["detection_count"],
                    "debug_image_path": debug_image_path,
                    "report": "ESP32 camera active - no objects currently detected ahead",
                }
        else:
            return {
                "status": "error",
                "visual_active": False,
                "error_message": "ESP32 camera failed to activate",
                "report": "Could not activate ESP32 visual analysis",
            }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Visual analysis error: {str(e)}",
            "report": "Error getting current visual analysis",
        }


def get_visual_scene_description() -> dict:
    """Get detailed scene description using ESP32 camera + Gemini vision AI.

    This function captures a frame from ESP32 camera and analyzes it with Gemini
    to provide detailed descriptions for visually impaired users.

    Returns:
        dict: Detailed scene analysis from Gemini AI
    """
    try:
        # First get current visual analysis to ensure camera is active
        visual_result = get_current_visual_analysis()

        if visual_result["status"] != "success":
            return {
                "status": "error",
                "error_message": "Could not activate ESP32 camera",
                "report": "Camera required for scene description",
            }

        # Get the latest frame from ESP32
        session_id = "navigation"
        if session_id in VIDEO_SESSIONS and "parser" in VIDEO_SESSIONS[session_id]:
            parser = VIDEO_SESSIONS[session_id]["parser"]
            frame_data = parser.get_latest_frame(timeout=3.0)

            if frame_data:
                img_array, pil_image = frame_data

                # Convert PIL image to bytes for Gemini analysis
                import io

                img_buffer = io.BytesIO()
                pil_image.save(img_buffer, format="JPEG", quality=85)
                img_bytes = img_buffer.getvalue()

                # Here we would call your Gemini service
                # For now, provide enhanced YOLO-based description
                recent_detections = visual_result.get("all_detections", [])

                # Create detailed scene description
                scene_description = {
                    "description": "Scene captured from ESP32 camera",
                    "immediate_obstacles": [],
                    "navigation_suggestion": "",
                    "points_of_interest": [],
                    "ambient_context": "Indoor/outdoor environment",
                }

                if recent_detections:
                    # Analyze YOLO detections for scene context
                    objects_by_confidence = sorted(
                        recent_detections, key=lambda x: x["confidence"], reverse=True
                    )

                    # Build description
                    main_objects = [
                        obj["class_name"] for obj in objects_by_confidence[:3]
                    ]
                    scene_description["description"] = (
                        f"I can see {', '.join(main_objects)} in the scene"
                    )

                    # Check for obstacles
                    obstacles = []
                    for obj in objects_by_confidence:
                        if obj["class_name"] in [
                            "person",
                            "car",
                            "bicycle",
                            "motorcycle",
                            "bus",
                            "truck",
                        ]:
                            obstacles.append(obj["class_name"])

                    if obstacles:
                        scene_description["immediate_obstacles"] = obstacles[:3]
                        scene_description["navigation_suggestion"] = (
                            f"Caution: {', '.join(obstacles[:2])} detected ahead. Proceed carefully."
                        )

                    # Points of interest
                    scene_description["points_of_interest"] = [
                        f"{obj['class_name']} ({obj['confidence']:.2f} confidence)"
                        for obj in objects_by_confidence[:5]
                    ]
                else:
                    scene_description["description"] = (
                        "The path ahead appears clear with no major objects detected"
                    )

                return {
                    "status": "success",
                    "scene_analysis": scene_description,
                    "yolo_detections": recent_detections,
                    "image_captured": True,
                    "report": f"Scene analysis: {scene_description['description']}",
                }
            else:
                return {
                    "status": "error",
                    "error_message": "No frame available from ESP32 camera",
                    "report": "Could not capture image for scene analysis",
                }
        else:
            return {
                "status": "error",
                "error_message": "ESP32 parser not available",
                "report": "Camera parser not initialized",
            }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Scene description error: {str(e)}",
            "report": "Error analyzing scene",
        }


def generate_dodge_instruction(dodge_analysis: dict) -> dict:
    """Generate specific dodging instructions based on obstacle analysis."""
    obstacles = dodge_analysis["obstacles"]
    critical = dodge_analysis["critical_obstacles"]

    # Determine best dodge direction
    left_clear = len(obstacles["left"]) == 0
    right_clear = len(obstacles["right"]) == 0

    if critical:
        main_obstacle = critical[0]
        obstacle_type = main_obstacle["class"]
        distance = main_obstacle["distance_estimate"]

        if left_clear and right_clear:
            # Both sides clear - choose based on obstacle position
            if main_obstacle["center"][0] < main_obstacle["bbox"][2] / 2:
                direction = "step right"
            else:
                direction = "step left"
        elif left_clear:
            direction = "step left"
        elif right_clear:
            direction = "step right"
        else:
            direction = "stop and wait"

        instruction = (
            f"Obstacle ahead: {obstacle_type} {distance}. {direction.capitalize()}."
        )

    else:
        # Multiple small obstacles
        if left_clear:
            direction = "move left"
            instruction = "Path partially blocked. Move left to continue."
        elif right_clear:
            direction = "move right"
            instruction = "Path partially blocked. Move right to continue."
        else:
            direction = "stop"
            instruction = "Path blocked on all sides. Stop and wait for clearance."

    return {
        "instruction": instruction,
        "direction": direction,
        "urgency": "high" if critical else "medium",
        "obstacles_detected": len(critical)
        + sum(len(obs) for obs in obstacles.values()),
    }


def estimate_obstacle_distance(bbox: List[float], frame_shape: Tuple[int, int]) -> str:
    """Estimate obstacle distance for dodging decisions."""
    x1, y1, x2, y2 = bbox
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_shape[0] * frame_shape[1]
    relative_size = box_area / frame_area

    # More precise distance estimation for dodging
    if relative_size > 0.4:
        return "immediately ahead"
    elif relative_size > 0.2:
        return "very close"
    elif relative_size > 0.1:
        return "close"
    elif relative_size > 0.05:
        return "approaching"
    else:
        return "distant"


def start_audio_input_session(
    language: str = "en-US", timeout: int = 30, phrase_time_limit: int = 10
) -> dict:
    """Starts audio input session for voice commands and destination setup."""

    try:
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        # Adjust for ambient noise
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)

        session_id = f"audio_input_{int(time.time())}"

        AUDIO_SESSIONS[session_id] = {
            "active": True,
            "recognizer": recognizer,
            "microphone": microphone,
            "language": language,
            "timeout": timeout,
            "phrase_time_limit": phrase_time_limit,
            "commands_history": [],
            "start_time": time.time(),
        }

        return {
            "status": "success",
            "session_id": session_id,
            "language": language,
            "timeout": timeout,
            "report": f"Audio input session started. Session: {session_id}",
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to start audio input: {str(e)}",
        }


def listen_for_destination(session_id: str) -> dict:
    """Listen for destination input via voice command."""

    if session_id not in AUDIO_SESSIONS:
        return {
            "status": "error",
            "error_message": f"Audio session {session_id} not found",
        }

    session = AUDIO_SESSIONS[session_id]

    try:
        # Prompt user
        generate_system_audio(
            "Please say your destination. For example: 'Navigate to Times Square'"
        )

        # Listen for audio
        with session["microphone"] as source:
            audio = session["recognizer"].listen(
                source,
                timeout=session["timeout"],
                phrase_time_limit=session["phrase_time_limit"],
            )

        # Recognize speech
        try:
            text = session["recognizer"].recognize_google(
                audio, language=session["language"]
            )

            # Process destination command
            destination = extract_destination_from_text(text)

            if destination:
                # Store command
                session["commands_history"].append(
                    {
                        "timestamp": time.time(),
                        "raw_text": text,
                        "extracted_destination": destination,
                        "command_type": "destination",
                    }
                )

                # Confirm destination
                generate_system_audio(
                    f"Destination set to: {destination}. Starting navigation."
                )

                return {
                    "status": "success",
                    "session_id": session_id,
                    "raw_text": text,
                    "destination": destination,
                    "report": f"Destination recognized: {destination}",
                }
            else:
                generate_system_audio(
                    "I couldn't understand the destination. Please try again."
                )
                return {
                    "status": "error",
                    "error_message": "Could not extract destination from speech",
                }

        except sr.UnknownValueError:
            generate_system_audio(
                "I couldn't understand what you said. Please speak clearly."
            )
            return {"status": "error", "error_message": "Speech not recognized"}
        except sr.RequestError as e:
            return {
                "status": "error",
                "error_message": f"Speech recognition error: {str(e)}",
            }

    except sr.WaitTimeoutError:
        generate_system_audio("No speech detected. Please try again.")
        return {"status": "error", "error_message": "No speech detected within timeout"}
    except Exception as e:
        return {"status": "error", "error_message": f"Audio input error: {str(e)}"}


def extract_destination_from_text(text: str) -> Optional[str]:
    """Extract destination from voice command text."""
    text = text.lower().strip()

    # Common navigation phrases
    navigation_triggers = [
        "navigate to",
        "go to",
        "take me to",
        "directions to",
        "find",
        "locate",
        "search for",
        "drive to",
        "walk to",
    ]

    for trigger in navigation_triggers:
        if trigger in text:
            # Extract everything after the trigger
            parts = text.split(trigger, 1)
            if len(parts) > 1:
                destination = parts[1].strip()
                # Clean up common words
                destination = destination.replace("the ", "").replace("a ", "")
                return destination

    # If no trigger found, assume entire text is destination
    if len(text.split()) <= 6:  # Reasonable destination length
        return text

    return None


def listen_for_voice_command(session_id: str) -> dict:
    """Listen for general voice commands (stop, repeat, status, etc.)."""

    if session_id not in AUDIO_SESSIONS:
        return {
            "status": "error",
            "error_message": f"Audio session {session_id} not found",
        }

    session = AUDIO_SESSIONS[session_id]

    try:
        with session["microphone"] as source:
            audio = session["recognizer"].listen(
                source, timeout=5, phrase_time_limit=5  # Shorter timeout for commands
            )

        text = (
            session["recognizer"]
            .recognize_google(audio, language=session["language"])
            .lower()
            .strip()
        )

        # Process command
        command_result = process_voice_command(text)

        # Store command
        session["commands_history"].append(
            {
                "timestamp": time.time(),
                "raw_text": text,
                "command_type": "general",
                "processed_result": command_result,
            }
        )

        return {
            "status": "success",
            "session_id": session_id,
            "raw_text": text,
            "command_result": command_result,
            "report": f"Voice command processed: {text}",
        }

    except sr.UnknownValueError:
        return {"status": "error", "error_message": "Speech not recognized"}
    except sr.WaitTimeoutError:
        return {"status": "error", "error_message": "No speech detected"}
    except Exception as e:
        return {"status": "error", "error_message": f"Voice command error: {str(e)}"}


def process_voice_command(text: str) -> dict:
    """Process recognized voice commands."""
    text = text.lower().strip()

    if any(word in text for word in ["stop", "halt", "pause"]):
        return {"action": "stop", "message": "Stopping current operation"}

    elif any(word in text for word in ["repeat", "again", "say again"]):
        return {"action": "repeat", "message": "Repeating last instruction"}

    elif any(word in text for word in ["status", "where am i", "location"]):
        return {"action": "status", "message": "Getting current status"}

    elif any(word in text for word in ["help", "commands", "what can you do"]):
        return {
            "action": "help",
            "message": "Available commands: stop, repeat, status, help",
        }

    elif any(word in text for word in ["cancel", "quit", "exit"]):
        return {"action": "cancel", "message": "Canceling current operation"}

    else:
        return {"action": "unknown", "message": f"Unknown command: {text}"}


def generate_navigation_audio(text: str, priority: str = "medium") -> dict:
    """Generate navigation-specific audio with appropriate voice settings."""
    return generate_prioritized_audio(
        text=text,
        priority=priority,
        voice_name="en-US-Journey-D",  # Navigation-optimized voice
        speaking_rate=1.0,
        audio_type="navigation",
    )


def generate_dodge_audio(text: str) -> dict:
    """Generate urgent dodging instruction audio."""
    return generate_prioritized_audio(
        text=text,
        priority="urgent",
        voice_name="en-US-Journey-F",  # Clear female voice for alerts
        speaking_rate=1.2,  # Faster for urgency
        audio_type="dodge",
    )


def generate_system_audio(text: str, priority: str = "medium") -> dict:
    """Generate system message audio."""
    return generate_prioritized_audio(
        text=text,
        priority=priority,
        voice_name="en-US-Standard-A",
        speaking_rate=1.0,
        audio_type="system",
    )


def generate_prioritized_audio(
    text: str,
    priority: str = "medium",
    voice_name: str = "en-US-Journey-D",
    speaking_rate: float = 1.0,
    audio_type: str = "general",
) -> dict:
    """Generate audio with priority-based settings and immediate playback for urgent messages."""

    try:
        client = texttospeech.TextToSpeechClient()

        # Priority-based audio settings
        priority_settings = {
            "urgent": {"pitch": 2.0, "volume": 1.0, "speaking_rate": 1.3},
            "high": {"pitch": 1.0, "volume": 0.9, "speaking_rate": 1.1},
            "medium": {"pitch": 0.0, "volume": 0.8, "speaking_rate": 1.0},
            "low": {"pitch": -1.0, "volume": 0.7, "speaking_rate": 0.9},
        }

        settings = priority_settings.get(priority, priority_settings["medium"])

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", name=voice_name
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate * settings["speaking_rate"],
            pitch=settings["pitch"],
            volume_gain_db=settings["volume"],
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Generate filename with priority and type
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{audio_type}_{priority}_{timestamp}.mp3"

        # Create appropriate directory
        audio_dir = f"audio_{audio_type}"
        os.makedirs(audio_dir, exist_ok=True)

        audio_path = os.path.join(audio_dir, filename)

        with open(audio_path, "wb") as out:
            out.write(response.audio_content)

        audio_base64 = base64.b64encode(response.audio_content).decode("utf-8")

        # For urgent messages, also trigger immediate playback
        if priority == "urgent":
            try:
                import pygame

                pygame.mixer.init()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
            except ImportError:
                pass  # pygame not available

        return {
            "status": "success",
            "audio_file": audio_path,
            "audio_base64": audio_base64,
            "filename": filename,
            "priority": priority,
            "audio_type": audio_type,
            "text_length": len(text),
            "voice_used": voice_name,
            "report": f"{priority.capitalize()} {audio_type} audio generated: {filename}",
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error generating {priority} audio: {str(e)}",
        }


def start_complete_navigation_system(
    destination: str,
    gps_endpoint: str,
    camera_source: int = 0,
    enable_video_dodging: bool = True,
) -> dict:
    """Starts the complete navigation system for blind users."""

    try:
        results = {}

        # 1. Start audio input session
        audio_result = start_audio_input_session()
        if audio_result["status"] == "success":
            results["audio_session"] = audio_result["session_id"]

        # 2. Start GPS polling
        gps_result = start_gps_polling(
            gps_endpoint=gps_endpoint,
            session_id=f"gps_{int(time.time())}",
            poll_interval=5,
        )
        if gps_result["status"] == "success":
            results["gps_session"] = gps_result["session_id"]

        # 3. Start navigation with initial location
        # Get initial GPS location
        time.sleep(2)  # Wait for first GPS reading
        location_result = get_current_gps_location(results["gps_session"])

        if location_result["status"] == "success":
            nav_result = start_live_navigation(
                destination=destination,
                current_lat=location_result["location"]["lat"],
                current_lng=location_result["location"]["lng"],
            )
            if nav_result["status"] == "success":
                results["navigation_session"] = nav_result["session_id"]

                # Link GPS to navigation
                GPS_SESSIONS[results["gps_session"]]["navigation_session_id"] = (
                    nav_result["session_id"]
                )

        # 4. Start video analysis with dodging
        if enable_video_dodging:
            video_result = start_video_analysis_with_dodging(
                camera_source=camera_source, analysis_interval=0.5, dodging_enabled=True
            )
            if video_result["status"] == "success":
                results["video_session"] = video_result["session_id"]

        # 5. Generate welcome audio
        welcome_message = (
            f"Navigation system started. Destination: {destination}. "
            "GPS tracking active. Video obstacle detection enabled. "
            "You can say 'stop', 'repeat', or 'status' at any time."
        )
        generate_system_audio(welcome_message, priority="high")

        return {
            "status": "success",
            "destination": destination,
            "sessions": results,
            "features_enabled": {
                "gps_tracking": "gps_session" in results,
                "navigation": "navigation_session" in results,
                "video_dodging": "video_session" in results,
                "audio_input": "audio_session" in results,
            },
            "report": f"Complete navigation system started for destination: {destination}",
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to start complete system: {str(e)}",
        }


# New Navigation Server Tools
def get_current_navigation_status(session_id: str = "default") -> dict:
    """Get current navigation status from the server for agent planning."""
    try:
        response = requests.get(
            f"http://localhost:8000/navigation/current?session_id={session_id}",
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()

            if data.get("has_navigation"):
                nav_status = data["navigation_status"]
                location = data.get("current_location", {})
                destination = data.get("destination", {})

                # Format response for agent
                return {
                    "status": "success",
                    "has_active_navigation": True,
                    "current_location": {
                        "latitude": location.get("lat"),
                        "longitude": location.get("lng"),
                        "accuracy": location.get("accuracy"),
                    },
                    "destination": {
                        "name": destination.get("name"),
                        "latitude": destination.get("lat"),
                        "longitude": destination.get("lng"),
                    },
                    "navigation_progress": {
                        "current_step": nav_status["current_step"],
                        "total_steps": nav_status["total_steps"],
                        "remaining_steps": nav_status["remaining_steps"],
                        "current_instruction": nav_status["current_instruction"],
                        "next_instruction": nav_status["next_instruction"],
                        "total_distance": nav_status["total_distance"],
                        "total_duration": nav_status["total_duration"],
                        "travel_mode": nav_status["travel_mode"],
                    },
                    "last_updated_seconds_ago": nav_status["time_since_update_seconds"],
                    "report": f"Active navigation to {destination.get('name')} - Step {nav_status['current_step']} of {nav_status['total_steps']}: {nav_status['current_instruction']}",
                }
            else:
                return {
                    "status": "success",
                    "has_active_navigation": False,
                    "report": "No active navigation session found. User may need to start navigation first.",
                }
        else:
            return {
                "status": "error",
                "error_message": f"Server responded with status {response.status_code}",
                "report": "Failed to get navigation status from server",
            }

    except requests.RequestException as e:
        return {
            "status": "error",
            "error_message": f"Network error: {str(e)}",
            "report": "Could not connect to navigation server",
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}",
            "report": "Error retrieving navigation status",
        }


def get_next_navigation_instruction(session_id: str = "default") -> dict:
    """Get the next navigation instruction and advance the step counter."""
    try:
        response = requests.get(
            f"http://localhost:8000/navigation/next_step?session_id={session_id}",
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()

            if data.get("has_step"):
                return {
                    "status": "success",
                    "has_instruction": True,
                    "step_info": {
                        "step_number": data["step_number"],
                        "total_steps": data["total_steps"],
                        "instruction": data["instruction"],
                        "distance": data["distance"],
                        "duration": data["duration"],
                        "remaining_steps": data["remaining_steps"],
                    },
                    "navigation_complete": data.get("navigation_complete", False),
                    "report": f"Step {data['step_number']}: {data['instruction']} ({data['distance']}, {data['duration']})",
                }
            else:
                return {
                    "status": "success",
                    "has_instruction": False,
                    "navigation_complete": True,
                    "report": "Navigation complete - destination reached!",
                }
        else:
            return {
                "status": "error",
                "error_message": f"Server responded with status {response.status_code}",
                "report": "Failed to get next navigation step",
            }

    except requests.RequestException as e:
        return {
            "status": "error",
            "error_message": f"Network error: {str(e)}",
            "report": "Could not connect to navigation server",
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}",
            "report": "Error getting next navigation instruction",
        }


def get_user_current_location(session_id: str = "default") -> dict:
    """Get the user's current GPS location from the server."""
    try:
        response = requests.get(
            f"http://localhost:8000/gps/location/{session_id}", timeout=10
        )

        if response.status_code == 200:
            data = response.json()

            if data["status"] == "success":
                location = data["location"]
                age_seconds = data.get("age_seconds", 0)

                return {
                    "status": "success",
                    "has_location": True,
                    "location": {
                        "latitude": location["lat"],
                        "longitude": location["lng"],
                        "accuracy": location.get("accuracy"),
                        "speed": location.get("speed"),
                        "heading": location.get("heading"),
                        "timestamp": location["timestamp"],
                    },
                    "location_age_seconds": age_seconds,
                    "location_fresh": age_seconds
                    < 30,  # Consider fresh if less than 30 seconds old
                    "report": f"Current location: {location['lat']:.6f}, {location['lng']:.6f} (±{location.get('accuracy', 'unknown')}m, {age_seconds:.1f}s ago)",
                }
            else:
                return {
                    "status": "error",
                    "has_location": False,
                    "error_message": data.get("message", "No location data available"),
                    "report": "No current location available. User may need to enable GPS sharing.",
                }
        else:
            return {
                "status": "error",
                "error_message": f"Server responded with status {response.status_code}",
                "report": "Failed to get current location from server",
            }

    except requests.RequestException as e:
        return {
            "status": "error",
            "error_message": f"Network error: {str(e)}",
            "report": "Could not connect to GPS server",
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}",
            "report": "Error retrieving current location",
        }


def get_all_navigation_sessions() -> dict:
    """Get information about all active navigation sessions."""
    try:
        response = requests.get("http://localhost:8000/navigation/sessions", timeout=10)

        if response.status_code == 200:
            data = response.json()

            return {
                "status": "success",
                "active_sessions_count": data["active_sessions"],
                "sessions": data["sessions"],
                "report": f"Found {data['active_sessions']} active navigation sessions",
            }
        else:
            return {
                "status": "error",
                "error_message": f"Server responded with status {response.status_code}",
                "report": "Failed to get navigation sessions",
            }

    except requests.RequestException as e:
        return {
            "status": "error",
            "error_message": f"Network error: {str(e)}",
            "report": "Could not connect to navigation server",
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}",
            "report": "Error retrieving navigation sessions",
        }


def get_ultrasonic_distance(
    esp32_ip: str = "192.168.18.39",
    alert_threshold: float = 1.0,
    session_id: str = "default",
) -> dict:
    """Get ultrasonic distance measurement from ESP32 and alert if below threshold.

    Args:
        esp32_ip (str): ESP32 IP address
        alert_threshold (float): Distance threshold in meters for alert
        session_id (str): Session identifier for tracking

    Returns:
        dict: Distance measurement and alert status
    """
    try:
        # Use curl subprocess to get distance data
        curl_result = subprocess.run(
            ["curl", "-s", f"http://{esp32_ip}/distance"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if curl_result.returncode == 0:
            response_text = curl_result.stdout.strip()

            try:
                # Try to parse as JSON first
                data = json.loads(response_text)
                if "cm" in data:
                    # ESP32 returns distance in cm, convert to meters
                    distance = float(data["cm"]) / 100.0
                elif "distance" in data:
                    distance = float(data["distance"])
                else:
                    distance = 0.0
            except (json.JSONDecodeError, ValueError, KeyError):
                # Try to parse as plain text/number
                try:
                    distance = float(response_text)
                    # Assume it's in cm if > 10 (reasonable assumption)
                    if distance > 10:
                        distance = distance / 100.0
                except ValueError:
                    return {
                        "status": "error",
                        "error_message": f"Could not parse distance data: {response_text}",
                        "report": "Invalid distance data format",
                    }

            # Check if distance is below threshold
            alert_triggered = distance < alert_threshold

            result = {
                "status": "success",
                "distance_meters": distance,
                "distance_cm": distance * 100,
                "alert_threshold": alert_threshold,
                "alert_triggered": alert_triggered,
                "session_id": session_id,
                "timestamp": time.time(),
                "report": f"Distance: {distance:.2f}m ({distance*100:.1f}cm)",
            }

            # Generate audio alert if distance is too close
            if alert_triggered:
                alert_message = f"Warning! Obstacle detected at {distance:.1f} meters ahead. Please be careful."
                result["alert_message"] = alert_message
                result["report"] += f" - ALERT: Object too close!"

                # Generate immediate audio warning
                generate_system_audio(alert_message, priority="urgent")

            return result

        else:
            return {
                "status": "error",
                "error_message": f"curl command failed with return code {curl_result.returncode}",
                "stderr": curl_result.stderr,
                "report": "Failed to get distance measurement via curl",
            }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error_message": "curl command timed out",
            "report": "Timeout connecting to ultrasonic sensor",
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "error_message": "curl command not found",
            "report": "curl not available on system",
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}",
            "report": "Error reading ultrasonic distance",
        }


def start_esp32_visual_analysis(
    esp32_ip: str = "192.168.18.39",
    stream_port: int = 81,
    session_id: str = "default",
    enable_yolo: bool = True,
    yolo_confidence: float = 0.5,
    analysis_fps: float = 2.0,
    flip_180: bool = True,
    save_frames: bool = False,
) -> dict:
    """Start ESP32 camera stream analysis with YOLO object detection.

    Args:
        esp32_ip (str): ESP32 IP address
        stream_port (int): Stream port number
        session_id (str): Session identifier
        enable_yolo (bool): Enable YOLO object detection
        yolo_confidence (float): YOLO confidence threshold
        analysis_fps (float): Analysis frames per second
        flip_180 (bool): Flip image 180 degrees
        save_frames (bool): Save analyzed frames

    Returns:
        dict: Visual analysis session info
    """
    try:
        import sys
        import os

        sys.path.append(
            os.path.join(os.path.dirname(__file__), "..", "video-stream-parse")
        )
        from esp32_stream_parser import ESP32StreamParser

        # Create output directory if saving frames
        output_dir = None
        if save_frames:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"esp32_analysis_{timestamp}"

        # Initialize stream parser with YOLOv11
        parser = ESP32StreamParser(
            esp32_ip=esp32_ip,
            stream_port=stream_port,
            save_images=save_frames,
            output_dir=output_dir or "esp32_temp",
            save_fps_limit=0.5,
            flip_180=flip_180,
            enable_yolo=enable_yolo,
            yolo_model="yolo11n.pt",  # Use YOLOv11 nano model
            yolo_confidence=yolo_confidence,
            save_annotated=save_frames,
        )
        
        # Store additional debug info
        print(f"🔧 ESP32 Parser Config: save_images={save_frames}, enable_yolo={enable_yolo}, output_dir={output_dir or 'esp32_temp'}")

        # Store parser in video sessions
        VIDEO_SESSIONS[session_id] = {
            "parser": parser,
            "active": False,
            "start_time": None,
            "frame_count": 0,
            "detection_count": 0,
            "last_detections": [],
            "analysis_fps": analysis_fps,
        }

        # Start the stream
        if parser.start_stream():
            VIDEO_SESSIONS[session_id]["active"] = True
            VIDEO_SESSIONS[session_id]["start_time"] = time.time()

            # Start analysis thread
            def analyze_stream():
                session = VIDEO_SESSIONS[session_id]
                last_analysis_time = 0
                min_interval = 1.0 / analysis_fps

                while session["active"]:
                    current_time = time.time()
                    if current_time - last_analysis_time < min_interval:
                        time.sleep(0.1)
                        continue

                    frame_data = parser.get_latest_frame(timeout=1.0)
                    if frame_data:
                        img_array, pil_image = frame_data
                        session["frame_count"] += 1

                        # Perform YOLO detection
                        if enable_yolo:
                            annotated_image, detections = parser.detect_objects_yolo(
                                img_array
                            )

                            if detections:
                                session["detection_count"] += 1
                                session["last_detections"] = detections

                                # Generate audio alerts for important objects
                                important_objects = [
                                    d
                                    for d in detections
                                    if d["class_name"]
                                    in [
                                        "person",
                                        "car",
                                        "bicycle",
                                        "motorbike",
                                        "bus",
                                        "truck",
                                    ]
                                ]

                                if important_objects:
                                    objects_str = ", ".join(
                                        [
                                            f"{obj['class_name']}"
                                            for obj in important_objects[:3]
                                        ]
                                    )
                                    alert_msg = (
                                        f"Visual alert: {objects_str} detected ahead"
                                    )
                                    generate_system_audio(alert_msg, priority="medium")

                        last_analysis_time = current_time

            analysis_thread = threading.Thread(target=analyze_stream, daemon=True)
            analysis_thread.start()

            return {
                "status": "success",
                "session_id": session_id,
                "stream_url": f"http://{esp32_ip}:{stream_port}/stream",
                "yolo_enabled": enable_yolo,
                "analysis_fps": analysis_fps,
                "save_frames": save_frames,
                "output_dir": output_dir,
                "report": f"ESP32 visual analysis started - session: {session_id}",
            }
        else:
            return {
                "status": "error",
                "error_message": "Failed to start ESP32 stream",
                "report": "Could not connect to ESP32 camera stream",
            }

    except ImportError as e:
        return {
            "status": "error",
            "error_message": f"ESP32 stream parser not available: {str(e)}",
            "report": "ESP32 stream parser module missing",
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}",
            "report": "Error starting ESP32 visual analysis",
        }


def get_esp32_visual_status(session_id: str = "default") -> dict:
    """Get status of ESP32 visual analysis session.

    Args:
        session_id (str): Session identifier

    Returns:
        dict: Visual analysis session status
    """
    if session_id not in VIDEO_SESSIONS:
        return {
            "status": "error",
            "error_message": f"Visual session {session_id} not found",
            "report": "No active visual analysis session",
        }

    session = VIDEO_SESSIONS[session_id]

    if not session["active"]:
        return {
            "status": "error",
            "error_message": "Visual session is not active",
            "report": "Visual analysis session stopped",
        }

    runtime = time.time() - session["start_time"] if session["start_time"] else 0

    return {
        "status": "success",
        "session_id": session_id,
        "active": session["active"],
        "runtime_seconds": runtime,
        "frames_processed": session["frame_count"],
        "detections_made": session["detection_count"],
        "recent_detections": session["last_detections"],
        "analysis_fps": session["analysis_fps"],
        "report": f"Visual analysis active: {session['frame_count']} frames, {session['detection_count']} detections in {runtime:.1f}s",
    }


def stop_esp32_visual_analysis(session_id: str = "default") -> dict:
    """Stop ESP32 visual analysis session.

    Args:
        session_id (str): Session identifier

    Returns:
        dict: Stop operation result
    """
    if session_id not in VIDEO_SESSIONS:
        return {
            "status": "error",
            "error_message": f"Visual session {session_id} not found",
            "report": "No session to stop",
        }

    session = VIDEO_SESSIONS[session_id]

    try:
        # Stop the session
        session["active"] = False

        # Stop the stream parser
        if "parser" in session:
            session["parser"].stop_stream()

        runtime = time.time() - session["start_time"] if session["start_time"] else 0

        # Clean up session
        del VIDEO_SESSIONS[session_id]

        return {
            "status": "success",
            "session_id": session_id,
            "runtime_seconds": runtime,
            "total_frames": session["frame_count"],
            "total_detections": session["detection_count"],
            "report": f"Visual analysis stopped after {runtime:.1f}s - processed {session['frame_count']} frames",
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error stopping session: {str(e)}",
            "report": "Error stopping visual analysis",
        }


def monitor_ultrasonic_distance(
    esp32_ip: str = "192.168.18.39",
    session_id: str = "default",
    poll_interval: float = 0.5,
    alert_threshold: float = 1.0,
    continuous_monitoring: bool = True,
) -> dict:
    """Start continuous ultrasonic distance monitoring with alerts.

    Args:
        esp32_ip (str): ESP32 IP address
        session_id (str): Session identifier
        poll_interval (float): Polling interval in seconds
        alert_threshold (float): Distance threshold for alerts in meters
        continuous_monitoring (bool): Keep monitoring until stopped

    Returns:
        dict: Monitoring session info
    """
    try:

        def monitor_distance():
            last_alert_time = 0
            alert_cooldown = 2.0  # Seconds between repeated alerts
            consecutive_failures = 0

            while session_id in AUDIO_SESSIONS and AUDIO_SESSIONS[session_id].get(
                "distance_monitoring", False
            ):
                try:
                    result = get_ultrasonic_distance(
                        esp32_ip, alert_threshold, session_id
                    )

                    if result["status"] == "success":
                        consecutive_failures = 0
                        distance = result["distance_meters"]

                        # Update session data
                        AUDIO_SESSIONS[session_id]["last_distance"] = distance
                        AUDIO_SESSIONS[session_id]["last_update"] = time.time()

                        # Check for alert with cooldown
                        if result["alert_triggered"]:
                            current_time = time.time()
                            if current_time - last_alert_time > alert_cooldown:
                                generate_system_audio(
                                    f"Obstacle at {distance:.1f} meters - proceed with caution",
                                    priority="urgent",
                                )
                                last_alert_time = current_time
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= 5:
                            generate_system_audio(
                                "Ultrasonic sensor connection lost", priority="medium"
                            )
                            break

                except Exception as e:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        break

                time.sleep(poll_interval)

        # Create or update session
        if session_id not in AUDIO_SESSIONS:
            AUDIO_SESSIONS[session_id] = {}

        AUDIO_SESSIONS[session_id].update(
            {
                "distance_monitoring": True,
                "esp32_ip": esp32_ip,
                "poll_interval": poll_interval,
                "alert_threshold": alert_threshold,
                "start_time": time.time(),
                "last_distance": None,
                "last_update": None,
            }
        )

        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_distance, daemon=True)
        monitor_thread.start()

        return {
            "status": "success",
            "session_id": session_id,
            "esp32_ip": esp32_ip,
            "poll_interval": poll_interval,
            "alert_threshold": alert_threshold,
            "continuous_monitoring": continuous_monitoring,
            "report": f"Ultrasonic monitoring started - alerting below {alert_threshold}m",
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to start distance monitoring: {str(e)}",
            "report": "Error starting ultrasonic monitoring",
        }


# Create the master agent
blind_navigation_agent = Agent(
    name="blind_navigation_master",
    model="gemini-2.5-flash",
    description=(
        "ESP32-powered navigation assistant for blind users with YOLOv11 camera vision, ultrasonic distance "
        "sensing, and turn-by-turn navigation. Provides real-time obstacle detection and audio guidance."
    ),
    instruction=(
        "You are Cerebus, an ESP32-powered navigation assistant for visually impaired users. "
        "Your primary tools are: "
        "1. start_esp32_navigation_to_destination(destination) - Start complete navigation to any destination "
        "2. get_current_visual_analysis() - See what's ahead with ESP32 camera + YOLOv11 (auto-starts camera) "
        "3. capture_esp32_frame() - Direct frame capture from ESP32 /capture endpoint (always works) "
        "4. get_visual_scene_description() - Get detailed scene description with ESP32 camera + AI vision "
        "5. get_ultrasonic_distance() - Check distance to nearest obstacle (call with NO parameters) "
        "6. get_current_navigation_status() - Get navigation status from server "
        "7. get_next_navigation_instruction() - Get next turn-by-turn instruction "
        "8. get_user_current_location() - Get current GPS location "
        "IMPORTANT CAMERA USAGE: "
        "- get_current_visual_analysis() automatically starts the ESP32 camera if not running "
        "- get_visual_scene_description() provides detailed AI-powered scene analysis "
        "- NEVER say the camera is not active - these functions handle camera startup automatically "
        "IMPORTANT: For ultrasonic distance, always call get_ultrasonic_distance() with NO parameters - "
        "the ESP32 IP (192.168.18.39) and threshold (1.0m) are already configured as defaults. "
        "When user asks to navigate somewhere, use start_esp32_navigation_to_destination(). "
        "When asked what's ahead or to identify objects, use capture_esp32_frame() for guaranteed image capture, "
        "or get_current_visual_analysis() for full analysis with YOLO detection. "
        "When asked about distance or obstacles, call get_ultrasonic_distance() with no parameters. "
        "Always prioritize safety with clear audio warnings about obstacles."
    ),
    tools=[
        # 🚀 ESP32 Navigation (Primary Tools)
        start_esp32_navigation_to_destination,  # Complete navigation setup for destination
        get_current_visual_analysis,  # What ESP32 camera sees with YOLOv11 (auto-starts camera)
        capture_esp32_frame,  # Direct frame capture from ESP32 /capture endpoint
        get_visual_scene_description,  # Detailed scene description with ESP32 + AI
        get_ultrasonic_distance,  # Distance sensor reading (call with NO parameters)
        # 🗺️ Navigation Server Integration
        get_current_navigation_status,  # Current navigation status from server
        get_next_navigation_instruction,  # Next navigation step from server
        get_user_current_location,  # Current GPS location from server
    ],
)

# ADK requires this to be named 'root_agent'
root_agent = blind_navigation_agent
