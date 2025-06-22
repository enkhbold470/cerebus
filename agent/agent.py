import asyncio
import base64
import datetime
import json
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

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
    model_path: str = "yolov8n.pt",
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
                    dodge_analysis = analyze_frame_for_dodging(frame, session["model"])
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


def analyze_frame_for_dodging(frame: np.ndarray, model: YOLO) -> dict:
    """Analyze frame specifically for dodging requirements."""
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


# Create the master agent
blind_navigation_agent = Agent(
    name="blind_navigation_master",
    model="gemini-2.5-flash",
    description=(
        "Master navigation system for blind users with GPS tracking, video obstacle detection, "
        "voice commands, and comprehensive audio feedback."
    ),
    instruction=(
        "You are a comprehensive navigation assistant for visually impaired users called Cerebus. You coordinate GPS tracking, "
        "real-time navigation updates, video-based obstacle detection with dodging instructions, "
        "voice command processing, and multi-priority audio feedback. Your primary goal is to "
        "provide safe, reliable navigation with clear audio instructions and immediate hazard alerts."
    ),
    tools=[
        # GPS Tools
        start_gps_polling,
        get_current_gps_location,
        # Navigation Tools
        start_live_navigation,
        update_navigation_location,
        get_navigation_status,
        # Video Analysis Tools
        start_video_analysis_with_dodging,
        analyze_frame_for_dodging,
        # Audio Input Tools
        start_audio_input_session,
        listen_for_destination,
        listen_for_voice_command,
        # Audio Output Tools
        generate_navigation_audio,
        generate_dodge_audio,
        generate_system_audio,
        generate_prioritized_audio,
        # System Coordination
        start_complete_navigation_system,
    ],
)

# ADK requires this to be named 'root_agent'
root_agent = blind_navigation_agent
