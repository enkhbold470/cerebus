# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import json
import logging
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np

# Imports
import pyaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openwakeword.model import Model

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--chunk_size",
    help="How much audio (in number of samples) to predict on at once",
    type=int,
    default=1280,
    required=False,
)
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    default="hey_cerebus.onnx",  # Default to custom model
    required=False,
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite'",
    type=str,
    default="onnx",  # Default to onnx for custom model
    required=False,
)
parser.add_argument(
    "--light_ip",
    help="IP address for the light controller",
    type=str,
    default="192.168.18.39",
    required=False,
)
parser.add_argument(
    "--timeout",
    help="Timeout in seconds before turning off the light",
    type=int,
    default=5,
    required=False,
)

args = parser.parse_args()

# FastAPI app setup
app = FastAPI(title="Cerebus Wake Word & WebRTC Audio Server")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class AudioEvent:
    event_type: str
    data: dict
    timestamp: float
    client_id: str


@dataclass
class LocationData:
    lat: float
    lng: float
    accuracy: Optional[float] = None
    timestamp: float = 0
    speed: Optional[float] = None
    heading: Optional[float] = None


@dataclass
class RouteStep:
    instruction: str
    distance: str
    duration: str
    start_location: LocationData
    end_location: LocationData


@dataclass
class NavigationData:
    session_id: str
    current_location: LocationData
    destination: Optional[LocationData] = None
    destination_name: Optional[str] = None
    route_steps: List[RouteStep] = None
    current_step: int = 0
    total_distance: Optional[str] = None
    total_duration: Optional[str] = None
    eta: Optional[str] = None
    travel_mode: str = "WALKING"
    last_updated: float = 0


# Global variables for wake word detection
owwModel = None
mic_stream = None
audio_interface = None
wake_word_processor = None
last_detection_time = 0
detection_cooldown = 2.0


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.sse_connections: Dict[str, asyncio.Queue] = {}
        self.audio_processors: Dict[str, "AudioProcessor"] = {}
        # GPS and Navigation data storage
        self.navigation_sessions: Dict[str, NavigationData] = {}
        self.location_history: Dict[str, List[LocationData]] = {}

    async def connect_websocket(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id}")

    def disconnect_websocket(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.audio_processors:
            self.audio_processors[client_id].stop()
            del self.audio_processors[client_id]
        logger.info(f"WebSocket disconnected: {client_id}")

    async def connect_sse(self, client_id: str):
        event_queue = asyncio.Queue()
        self.sse_connections[client_id] = event_queue
        logger.info(f"SSE connected: {client_id}")
        return event_queue

    def disconnect_sse(self, client_id: str):
        if client_id in self.sse_connections:
            del self.sse_connections[client_id]
        logger.info(f"SSE disconnected: {client_id}")

    async def send_event_to_client(self, client_id: str, event: AudioEvent):
        if client_id in self.sse_connections:
            try:
                await self.sse_connections[client_id].put(event)
            except Exception as e:
                logger.error(f"Error sending event to {client_id}: {e}")

    async def broadcast_event(self, event: AudioEvent):
        for client_id in list(self.sse_connections.keys()):
            await self.send_event_to_client(client_id, event)


class AudioProcessor:
    def __init__(self, client_id: str, manager: ConnectionManager):
        self.client_id = client_id
        self.manager = manager
        self.audio_buffer = queue.Queue()
        self.is_running = False
        self.processing_thread = None
        self.wake_word_detected = False
        self.audio_level = 0.0
        self.loop = None  # Store reference to the main event loop
        self.chunks_received = 0  # Track number of audio chunks received

    def start(self):
        self.is_running = True
        # Store reference to the current event loop
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                f"No event loop running when starting AudioProcessor for {self.client_id}"
            )

        self.processing_thread = threading.Thread(target=self._process_audio_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info(f"🎵 Audio processor started for {self.client_id}")

    def stop(self):
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        logger.info(f"Audio processor stopped for {self.client_id}")

    def add_audio_data(self, audio_data: bytes):
        if self.is_running:
            self.chunks_received += 1
            self.audio_buffer.put(audio_data)

            # Log every 50 chunks to avoid spam
            if self.chunks_received % 50 == 0:
                logger.info(
                    f"📦 {self.client_id}: Received {self.chunks_received} audio chunks ({len(audio_data)} bytes)"
                )
            elif self.chunks_received <= 5:  # Log first few chunks
                logger.info(
                    f"📦 {self.client_id}: Audio chunk #{self.chunks_received} received ({len(audio_data)} bytes)"
                )

    def _process_audio_loop(self):
        while self.is_running:
            try:
                # Get audio data with timeout
                audio_data = self.audio_buffer.get(timeout=0.1)

                # Process the audio chunk in the main event loop
                if self.loop and not self.loop.is_closed():
                    try:
                        future = asyncio.run_coroutine_threadsafe(
                            self._process_audio_chunk(audio_data), self.loop
                        )
                        # Don't wait for completion to avoid blocking
                    except Exception as e:
                        logger.error(f"Error scheduling audio processing: {e}")
                else:
                    # Try to get the current loop as fallback
                    try:
                        loop = asyncio.get_event_loop()
                        asyncio.run_coroutine_threadsafe(
                            self._process_audio_chunk(audio_data), loop
                        )
                    except RuntimeError:
                        # If no event loop is available, skip this chunk
                        if self.chunks_received <= 10:  # Only warn for first 10 chunks
                            logger.warning(
                                f"⚠️ No event loop available for {self.client_id}, skipping audio chunk"
                            )
                        continue

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(
                    f"Error in audio processing loop for {self.client_id}: {e}"
                )

    async def _process_audio_chunk(self, audio_data: bytes):
        try:
            # Calculate audio level
            if len(audio_data) > 0:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                if len(audio_array) > 0:
                    self.audio_level = float(np.abs(audio_array).mean())

            # Log processing details every 100 chunks
            if self.chunks_received % 100 == 0:
                logger.info(
                    f"🔊 {self.client_id}: Processing chunk #{self.chunks_received}, audio level: {self.audio_level:.1f}"
                )

            # Run wake word detection if model is available
            wake_word_score = 0.0
            if owwModel is not None:
                wake_word_score = await self._detect_wake_word(audio_data)
            else:
                # Log if model is not available (only for first few chunks)
                if self.chunks_received <= 3:
                    logger.warning(
                        f"⚠️ {self.client_id}: Wake word model not available for processing"
                    )

            # Send audio level event
            await self.manager.send_event_to_client(
                self.client_id,
                AudioEvent(
                    event_type="audio_level",
                    data={"level": self.audio_level, "timestamp": time.time()},
                    timestamp=time.time(),
                    client_id=self.client_id,
                ),
            )

            # Send wake word detection result
            if wake_word_score > 0.5:  # Threshold for wake word detection
                global last_detection_time
                current_time = time.time()
                if (current_time - last_detection_time) > detection_cooldown:
                    self.wake_word_detected = True
                    last_detection_time = current_time

                    logger.info(
                        f"🔥 {self.client_id}: WAKE WORD DETECTED! Confidence: {wake_word_score:.3f}"
                    )

                    await self.manager.send_event_to_client(
                        self.client_id,
                        AudioEvent(
                            event_type="wake_word_detected",
                            data={
                                "detected": True,
                                "confidence": wake_word_score,
                                "timestamp": time.time(),
                            },
                            timestamp=time.time(),
                            client_id=self.client_id,
                        ),
                    )

                    # Send agent start message
                    await self.manager.send_event_to_client(
                        self.client_id,
                        AudioEvent(
                            event_type="agent_start",
                            data={
                                "message": "Cerebus agent is now listening...",
                                "confidence": wake_word_score,
                                "timestamp": time.time(),
                                "status": "active",
                            },
                            timestamp=time.time(),
                            client_id=self.client_id,
                        ),
                    )

                    # Trigger light control
                    await self._handle_wake_word_detection()

                    # Reset wake word detection after 3 seconds
                    await asyncio.sleep(3)
                    self.wake_word_detected = False
                else:
                    # Log when wake word is detected but cooldown is active
                    logger.debug(
                        f"🔥 {self.client_id}: Wake word detected (confidence: {wake_word_score:.3f}) but cooldown active"
                    )
            elif wake_word_score > 0.1:  # Log lower confidence scores occasionally
                if self.chunks_received % 200 == 0:
                    logger.debug(
                        f"🔍 {self.client_id}: Low confidence wake word score: {wake_word_score:.3f}"
                    )

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")

    async def _detect_wake_word(self, audio_data: bytes) -> float:
        """Run wake word detection using the loaded model"""
        try:
            if owwModel is None:
                return 0.0

            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Ensure we have enough samples for the model
            if len(audio_array) < args.chunk_size:
                return 0.0

            # Truncate or pad to expected chunk size
            if len(audio_array) > args.chunk_size:
                audio_array = audio_array[: args.chunk_size]
            elif len(audio_array) < args.chunk_size:
                audio_array = np.pad(
                    audio_array, (0, args.chunk_size - len(audio_array))
                )

            # Run prediction
            prediction = owwModel.predict(audio_array)

            # Get the highest confidence score
            max_score = 0.0
            for mdl in owwModel.prediction_buffer.keys():
                scores = list(owwModel.prediction_buffer[mdl])
                if scores:
                    max_score = max(max_score, scores[-1])

            return max_score

        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
            return 0.0

    async def _handle_wake_word_detection(self):
        """Handle wake word detection with light control"""
        logger.info("🔥 WAKE WORD DETECTED! Triggering light...")

        # Trigger light immediately
        await asyncio.get_event_loop().run_in_executor(
            None, trigger_light, args.light_ip
        )

        # Schedule light off after timeout
        async def delayed_light_off():
            logger.info(
                f"⏱️ Waiting {args.timeout} seconds (placeholder for agentic processing)..."
            )
            await asyncio.sleep(args.timeout)
            await asyncio.get_event_loop().run_in_executor(
                None, turn_off_light, args.light_ip
            )
            logger.info("💡 Light control sequence completed")

        # Run light off in background
        asyncio.create_task(delayed_light_off())


class WakeWordProcessor:
    def __init__(self):
        self.is_running = False
        self.processing_thread = None
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = args.chunk_size

    def start(self):
        """Start the wake word processor"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._microphone_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Wake word processor started")

    def stop(self):
        """Stop the wake word processor"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        logger.info("Wake word processor stopped")

    def _microphone_loop(self):
        """Main microphone processing loop"""
        global mic_stream, audio_interface, last_detection_time

        try:
            # Initialize audio
            audio_interface = pyaudio.PyAudio()
            mic_stream = audio_interface.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
            )

            print("\n" + "#" * 100)
            print("🎤 Microphone wake word detection active")
            print(f"Model: {args.model_path}")
            print(f"Framework: {args.inference_framework}")
            print(f"Light Controller: {args.light_ip}")
            print(f"Timeout: {args.timeout} seconds")
            print("#" * 100)

            while self.is_running:
                try:
                    # Get audio
                    audio_data = mic_stream.read(
                        self.CHUNK, exception_on_overflow=False
                    )
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)

                    # Feed to openWakeWord model
                    prediction = owwModel.predict(audio_array)

                    # Check for wake word detection
                    wake_word_detected = False
                    max_score = 0.0

                    for mdl in owwModel.prediction_buffer.keys():
                        scores = list(owwModel.prediction_buffer[mdl])
                        if scores and scores[-1] > 0.5:
                            wake_word_detected = True
                            max_score = max(max_score, scores[-1])

                    # Handle wake word detection with cooldown
                    current_time = time.time()
                    if (
                        wake_word_detected
                        and (current_time - last_detection_time) > detection_cooldown
                    ):
                        logger.info(
                            f"🔥 WAKE WORD DETECTED! Confidence: {max_score:.3f}"
                        )

                        # Trigger light control
                        trigger_light(args.light_ip)

                        # Broadcast wake word event to all connected clients
                        wake_word_event = AudioEvent(
                            event_type="wake_word_detected",
                            data={
                                "detected": True,
                                "confidence": max_score,
                                "timestamp": time.time(),
                                "source": "microphone",
                            },
                            timestamp=time.time(),
                            client_id="microphone",
                        )

                        # Broadcast agent start event to all connected clients
                        agent_start_event = AudioEvent(
                            event_type="agent_start",
                            data={
                                "message": "Cerebus agent is now listening...",
                                "confidence": max_score,
                                "timestamp": time.time(),
                                "status": "active",
                                "source": "microphone",
                            },
                            timestamp=time.time(),
                            client_id="microphone",
                        )

                        # Schedule broadcasts
                        try:
                            loop = asyncio.get_event_loop()
                            asyncio.run_coroutine_threadsafe(
                                manager.broadcast_event(wake_word_event), loop
                            )
                            asyncio.run_coroutine_threadsafe(
                                manager.broadcast_event(agent_start_event), loop
                            )
                        except RuntimeError:
                            logger.warning("No event loop available for broadcasting")

                        last_detection_time = current_time

                        # Schedule light off
                        def delayed_light_off():
                            time.sleep(args.timeout)
                            turn_off_light(args.light_ip)

                        threading.Thread(target=delayed_light_off, daemon=True).start()

                except Exception as e:
                    if self.is_running:
                        logger.error(f"Error in microphone loop: {e}")

        except Exception as e:
            logger.error(f"Error initializing microphone: {e}")
        finally:
            if mic_stream:
                mic_stream.close()
            if audio_interface:
                audio_interface.terminate()


# Global connection manager
manager = ConnectionManager()

# Resolve model path using pathlib
script_dir = Path(__file__).parent
if args.model_path and not Path(args.model_path).is_absolute():
    # If it's a relative path, resolve it relative to the script directory
    model_path = script_dir / args.model_path
    if not model_path.exists():
        # Also try current working directory
        cwd_model_path = Path.cwd() / args.model_path
        if cwd_model_path.exists():
            model_path = cwd_model_path
        else:
            print(f"⚠️ Model file not found at: {model_path}")
            print(f"⚠️ Also checked: {cwd_model_path}")
            print(f"💡 Available files in {script_dir}:")
            for file in script_dir.glob("*.onnx"):
                print(f"   - {file.name}")
    args.model_path = str(model_path)
else:
    model_path = Path(args.model_path) if args.model_path else None

print(f"🔍 Using model path: {args.model_path}")


# Light control functions
def trigger_light(ip_address):
    """Trigger the light flash"""
    try:
        url = f"http://{ip_address}/rgb/flash"
        result = subprocess.run(
            ["curl", "-s", "-m", "10", url], capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            print(f"✓ Light triggered: {url}")
            if result.stdout:
                print(f"   Response: {result.stdout.strip()}")
        else:
            print(f"⚠️ Light trigger failed: curl returned {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        print(f"✗ Light trigger timeout: {url}")
    except Exception as e:
        print(f"✗ Light trigger error: {e}")


def turn_off_light(ip_address):
    """Turn off the light"""
    try:
        url = f"http://{ip_address}/rgb/off"
        result = subprocess.run(
            ["curl", "-s", "-m", "10", url], capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            print(f"✓ Light turned off: {url}")
            if result.stdout:
                print(f"   Response: {result.stdout.strip()}")
        else:
            print(f"⚠️ Light off failed: curl returned {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        print(f"✗ Light off timeout: {url}")
    except Exception as e:
        print(f"✗ Light off error: {e}")


def handle_wake_word_detection(ip_address, timeout_seconds):
    """Handle wake word detection with light control"""
    print(f"\n🔥 WAKE WORD DETECTED! Triggering light...")

    # Trigger light immediately
    trigger_light(ip_address)

    # Run light off in separate thread to avoid blocking detection
    def delayed_light_off():
        time.sleep(timeout_seconds)
        turn_off_light(ip_address)
        print("💡 Light control sequence completed\n")

    threading.Thread(target=delayed_light_off, daemon=True).start()


# FastAPI Routes
@app.websocket("/ws/audio/{client_id}")
async def websocket_audio_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect_websocket(websocket, client_id)

    # Start audio processor
    audio_processor = AudioProcessor(client_id, manager)
    manager.audio_processors[client_id] = audio_processor
    audio_processor.start()

    logger.info(f"🎙️ WebSocket audio endpoint ready for {client_id}")
    bytes_received_total = 0

    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            bytes_received_total += len(data)

            # Add to audio processor
            audio_processor.add_audio_data(data)

            # Send acknowledgment
            await websocket.send_json(
                {"type": "ack", "timestamp": time.time(), "bytes_received": len(data)}
            )

    except WebSocketDisconnect:
        logger.info(
            f"🔌 {client_id} disconnected. Total bytes received: {bytes_received_total}"
        )
    finally:
        manager.disconnect_websocket(client_id)


# Initialize audio variables
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size

# Global variables - initialize later
owwModel = None
mic_stream = None
audio_interface = None
n_models = 0


@app.get("/")
async def root():
    return {
        "message": "Cerebus Wake Word & WebRTC Audio Server",
        "version": "1.0.0",
        "model": args.model_path,
        "light_ip": args.light_ip,
    }


@app.get("/status")
async def get_status():
    """Get server status and connected clients"""
    return {
        "status": "running",
        "active_websocket_connections": len(manager.active_connections),
        "active_sse_connections": len(manager.sse_connections),
        "active_audio_processors": len(manager.audio_processors),
        "connected_clients": list(manager.active_connections.keys()),
        "wake_word_model": args.model_path,
        "light_controller": args.light_ip,
        "microphone_active": (
            wake_word_processor.is_running if wake_word_processor else False
        ),
        "timestamp": time.time(),
    }


@app.post("/trigger_event/{client_id}")
async def trigger_event(client_id: str, event_data: dict):
    """Manually trigger an event for testing purposes"""
    event = AudioEvent(
        event_type="manual_trigger",
        data=event_data,
        timestamp=time.time(),
        client_id=client_id,
    )

    await manager.send_event_to_client(client_id, event)
    return {"message": f"Event sent to {client_id}", "event": event_data}


@app.post("/broadcast_event")
async def broadcast_event(event_data: dict):
    """Broadcast an event to all connected clients"""
    event = AudioEvent(
        event_type="broadcast",
        data=event_data,
        timestamp=time.time(),
        client_id="server",
    )

    await manager.broadcast_event(event)
    return {"message": "Event broadcasted to all clients", "event": event_data}


@app.post("/light/trigger")
async def trigger_light_endpoint():
    """Trigger the light manually"""
    await asyncio.get_event_loop().run_in_executor(None, trigger_light, args.light_ip)
    return {"message": "Light triggered", "ip": args.light_ip}


@app.post("/light/off")
async def turn_off_light_endpoint():
    """Turn off the light manually"""
    await asyncio.get_event_loop().run_in_executor(None, turn_off_light, args.light_ip)
    return {"message": "Light turned off", "ip": args.light_ip}


# GPS and Navigation Endpoints
@app.post("/gps/location")
async def update_location(location_data: dict):
    """Receive GPS location updates from frontend"""
    try:
        session_id = location_data.get("sessionId", "default")

        location = LocationData(
            lat=location_data["lat"],
            lng=location_data["lng"],
            accuracy=location_data.get("accuracy"),
            timestamp=location_data.get("timestamp", time.time()),
            speed=location_data.get("speed"),
            heading=location_data.get("heading"),
        )

        # Store in location history
        if session_id not in manager.location_history:
            manager.location_history[session_id] = []

        manager.location_history[session_id].append(location)

        # Keep only last 100 locations per session
        if len(manager.location_history[session_id]) > 100:
            manager.location_history[session_id] = manager.location_history[session_id][
                -100:
            ]

        # Update current location in navigation session if exists
        if session_id in manager.navigation_sessions:
            manager.navigation_sessions[session_id].current_location = location
            manager.navigation_sessions[session_id].last_updated = time.time()

        logger.info(
            f"📍 Location updated for {session_id}: {location.lat:.6f}, {location.lng:.6f}"
        )

        return {
            "status": "success",
            "message": "Location updated",
            "session_id": session_id,
            "location": {
                "lat": location.lat,
                "lng": location.lng,
                "accuracy": location.accuracy,
                "timestamp": location.timestamp,
            },
        }

    except Exception as e:
        logger.error(f"Error updating location: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/navigation/route")
async def update_navigation_route(route_data: dict):
    """Receive route/navigation data from frontend"""
    try:
        session_id = route_data.get("sessionId", "default")

        # Extract destination
        destination = None
        if "destination" in route_data:
            dest_data = route_data["destination"]
            destination = LocationData(
                lat=dest_data["lat"], lng=dest_data["lng"], timestamp=time.time()
            )

        # Extract current location
        current_location = None
        if "origin" in route_data:
            origin_data = route_data["origin"]
            current_location = LocationData(
                lat=origin_data["lat"], lng=origin_data["lng"], timestamp=time.time()
            )

        # Extract route steps
        route_steps = []
        if "detailed_steps" in route_data:
            for step_data in route_data["detailed_steps"]:
                step = RouteStep(
                    instruction=step_data["instruction"],
                    distance=step_data["distance"],
                    duration=step_data["duration"],
                    start_location=LocationData(
                        lat=step_data["start_location"]["lat"],
                        lng=step_data["start_location"]["lng"],
                        timestamp=time.time(),
                    ),
                    end_location=LocationData(
                        lat=step_data["end_location"]["lat"],
                        lng=step_data["end_location"]["lng"],
                        timestamp=time.time(),
                    ),
                )
                route_steps.append(step)

        # Create or update navigation session
        navigation_data = NavigationData(
            session_id=session_id,
            current_location=current_location,
            destination=destination,
            destination_name=route_data.get("destination_name"),
            route_steps=route_steps,
            total_distance=route_data.get("route_summary", {}).get("distance"),
            total_duration=route_data.get("route_summary", {}).get("duration"),
            travel_mode=route_data.get("route_summary", {}).get(
                "travel_mode", "WALKING"
            ),
            last_updated=time.time(),
        )

        manager.navigation_sessions[session_id] = navigation_data

        logger.info(
            f"🗺️ Navigation route updated for {session_id}: {len(route_steps)} steps to {route_data.get('destination_name', 'unknown destination')}"
        )

        return {
            "status": "success",
            "message": "Navigation route updated",
            "session_id": session_id,
            "steps_count": len(route_steps),
            "destination": route_data.get("destination_name"),
            "total_distance": navigation_data.total_distance,
            "total_duration": navigation_data.total_duration,
        }

    except Exception as e:
        logger.error(f"Error updating navigation route: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/navigation/current")
async def get_current_navigation(session_id: str = "default"):
    """Get current navigation data for agent consumption"""
    try:
        if session_id not in manager.navigation_sessions:
            return {
                "status": "error",
                "message": f"No navigation session found for {session_id}",
                "has_navigation": False,
            }

        nav_data = manager.navigation_sessions[session_id]

        # Get current step instruction
        current_instruction = None
        next_instruction = None
        remaining_steps = 0

        if nav_data.route_steps and nav_data.current_step < len(nav_data.route_steps):
            current_instruction = nav_data.route_steps[
                nav_data.current_step
            ].instruction
            remaining_steps = len(nav_data.route_steps) - nav_data.current_step

            if nav_data.current_step + 1 < len(nav_data.route_steps):
                next_instruction = nav_data.route_steps[
                    nav_data.current_step + 1
                ].instruction

        # Calculate time since last update
        time_since_update = time.time() - nav_data.last_updated

        result = {
            "status": "success",
            "has_navigation": True,
            "session_id": session_id,
            "current_location": (
                {
                    "lat": nav_data.current_location.lat,
                    "lng": nav_data.current_location.lng,
                    "accuracy": nav_data.current_location.accuracy,
                    "timestamp": nav_data.current_location.timestamp,
                }
                if nav_data.current_location
                else None
            ),
            "destination": (
                {
                    "lat": nav_data.destination.lat,
                    "lng": nav_data.destination.lng,
                    "name": nav_data.destination_name,
                }
                if nav_data.destination
                else None
            ),
            "navigation_status": {
                "current_step": nav_data.current_step,
                "total_steps": len(nav_data.route_steps) if nav_data.route_steps else 0,
                "remaining_steps": remaining_steps,
                "current_instruction": current_instruction,
                "next_instruction": next_instruction,
                "total_distance": nav_data.total_distance,
                "total_duration": nav_data.total_duration,
                "travel_mode": nav_data.travel_mode,
                "last_updated": nav_data.last_updated,
                "time_since_update_seconds": time_since_update,
            },
        }

        logger.info(
            f"🧭 Navigation data requested for {session_id} - {remaining_steps} steps remaining"
        )

        return result

    except Exception as e:
        logger.error(f"Error getting current navigation: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/navigation/next_step")
async def get_next_navigation_step(session_id: str = "default"):
    """Get next navigation step for agent guidance"""
    try:
        if session_id not in manager.navigation_sessions:
            return {
                "status": "error",
                "message": f"No navigation session found for {session_id}",
                "has_step": False,
            }

        nav_data = manager.navigation_sessions[session_id]

        if not nav_data.route_steps or nav_data.current_step >= len(
            nav_data.route_steps
        ):
            return {
                "status": "success",
                "message": "Navigation complete - no more steps",
                "has_step": False,
                "navigation_complete": True,
            }

        current_step = nav_data.route_steps[nav_data.current_step]

        # Advance to next step
        nav_data.current_step += 1
        nav_data.last_updated = time.time()

        return {
            "status": "success",
            "has_step": True,
            "step_number": nav_data.current_step,
            "total_steps": len(nav_data.route_steps),
            "instruction": current_step.instruction,
            "distance": current_step.distance,
            "duration": current_step.duration,
            "start_location": {
                "lat": current_step.start_location.lat,
                "lng": current_step.start_location.lng,
            },
            "end_location": {
                "lat": current_step.end_location.lat,
                "lng": current_step.end_location.lng,
            },
            "remaining_steps": len(nav_data.route_steps) - nav_data.current_step,
            "navigation_complete": nav_data.current_step >= len(nav_data.route_steps),
        }

    except Exception as e:
        logger.error(f"Error getting next navigation step: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/gps/location/{session_id}")
async def get_current_location(session_id: str):
    """Get current location for a session"""
    try:
        if (
            session_id in manager.location_history
            and manager.location_history[session_id]
        ):
            latest_location = manager.location_history[session_id][-1]
            return {
                "status": "success",
                "session_id": session_id,
                "location": {
                    "lat": latest_location.lat,
                    "lng": latest_location.lng,
                    "accuracy": latest_location.accuracy,
                    "timestamp": latest_location.timestamp,
                    "speed": latest_location.speed,
                    "heading": latest_location.heading,
                },
                "age_seconds": time.time() - latest_location.timestamp,
            }
        else:
            return {
                "status": "error",
                "message": f"No location data found for session {session_id}",
                "has_location": False,
            }

    except Exception as e:
        logger.error(f"Error getting current location: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/navigation/sessions")
async def get_navigation_sessions():
    """Get all active navigation sessions"""
    try:
        sessions = []
        for session_id, nav_data in manager.navigation_sessions.items():
            session_info = {
                "session_id": session_id,
                "destination_name": nav_data.destination_name,
                "current_step": nav_data.current_step,
                "total_steps": len(nav_data.route_steps) if nav_data.route_steps else 0,
                "last_updated": nav_data.last_updated,
                "time_since_update": time.time() - nav_data.last_updated,
                "travel_mode": nav_data.travel_mode,
                "has_current_location": nav_data.current_location is not None,
            }
            sessions.append(session_info)

        return {
            "status": "success",
            "active_sessions": len(sessions),
            "sessions": sessions,
        }

    except Exception as e:
        logger.error(f"Error getting navigation sessions: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/sse/{client_id}")
async def sse_endpoint(client_id: str):
    """Server-Sent Events endpoint for real-time communication with frontend"""

    async def event_generator():
        # Connect to SSE
        event_queue = await manager.connect_sse(client_id)

        try:
            # Send initial connection event
            initial_event = {
                "type": "connected",
                "data": {
                    "message": f"SSE connected for {client_id}",
                    "timestamp": time.time(),
                },
                "timestamp": time.time(),
                "client_id": client_id,
            }

            yield f"data: {json.dumps(initial_event)}\n\n"

            # Keep connection alive and send events
            while True:
                try:
                    # Wait for event with timeout for keepalive
                    event = await asyncio.wait_for(event_queue.get(), timeout=30.0)

                    # Send the event
                    event_data = {
                        "type": event.event_type,
                        "data": event.data,
                        "timestamp": event.timestamp,
                        "client_id": event.client_id,
                    }

                    yield f"data: {json.dumps(event_data)}\n\n"

                except asyncio.TimeoutError:
                    # Send keepalive
                    keepalive_event = {
                        "type": "keepalive",
                        "data": {"timestamp": time.time()},
                        "timestamp": time.time(),
                        "client_id": client_id,
                    }
                    yield f"data: {json.dumps(keepalive_event)}\n\n"

        except asyncio.CancelledError:
            logger.info(f"SSE connection cancelled for {client_id}")
        except Exception as e:
            logger.error(f"Error in SSE stream for {client_id}: {e}")
        finally:
            manager.disconnect_sse(client_id)
            logger.info(f"SSE disconnected for {client_id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )


async def timestamp_broadcaster():
    """Background task to broadcast timestamp every second for testing"""
    while True:
        try:
            await asyncio.sleep(1.0)  # Wait 1 second

            # Create timestamp event
            timestamp_event = AudioEvent(
                event_type="timestamp_test",
                data={
                    "timestamp": time.time(),
                    "datetime": datetime.now().isoformat(),
                    "message": "Test timestamp broadcast",
                },
                timestamp=time.time(),
                client_id="server",
            )

            # Broadcast to all connected SSE clients
            await manager.broadcast_event(timestamp_event)

        except Exception as e:
            logger.error(f"Error in timestamp broadcaster: {e}")
            await asyncio.sleep(1.0)  # Continue after error


@app.on_event("startup")
async def startup_event():
    """Initialize the wake word model and start microphone processing"""
    global owwModel, wake_word_processor, n_models

    # Load pre-trained openwakeword models
    try:
        if args.model_path != "":
            owwModel = Model(
                wakeword_models=[args.model_path],
                inference_framework=args.inference_framework,
            )
        else:
            owwModel = Model(inference_framework=args.inference_framework)

        n_models = len(owwModel.models.keys())
        logger.info(f"✓ Wake word model loaded: {args.model_path}")
        logger.info(f"✓ Available models: {list(owwModel.models.keys())}")

        # Start microphone-based wake word detection
        wake_word_processor = WakeWordProcessor()
        wake_word_processor.start()

        # Start timestamp broadcaster for testing
        asyncio.create_task(timestamp_broadcaster())
        logger.info("✓ Timestamp broadcaster started (1-second interval)")

    except Exception as e:
        logger.error(f"✗ Error loading wake word model: {e}")
        owwModel = None
        n_models = 0


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources"""
    global wake_word_processor, mic_stream, audio_interface

    if wake_word_processor:
        wake_word_processor.stop()

    if mic_stream:
        mic_stream.close()

    if audio_interface:
        audio_interface.terminate()


if __name__ == "__main__":
    # Check if we want to run the standalone microphone detection
    import sys

    import uvicorn

    if "--standalone" in sys.argv:
        # Initialize model and microphone for standalone mode
        try:
            if args.model_path != "":
                owwModel = Model(
                    wakeword_models=[args.model_path],
                    inference_framework=args.inference_framework,
                )
            else:
                owwModel = Model(inference_framework=args.inference_framework)

            n_models = len(owwModel.models.keys())

            # Initialize audio
            audio = pyaudio.PyAudio()
            mic_stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )

            # Generate output string header
            print("\n\n")
            print("#" * 100)
            print("Listening for wakewords...")
            print(f"Model: {args.model_path}")
            print(f"Framework: {args.inference_framework}")
            print(f"Light Controller: {args.light_ip}")
            print(f"Timeout: {args.timeout} seconds")
            print("#" * 100)
            print("\n" * (n_models * 3))

            # Track wake word detection state
            last_detection_time = 0
            detection_cooldown = 2.0  # Prevent multiple triggers within 2 seconds

            try:
                while True:
                    # Get audio
                    audio_data = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

                    # Feed to openWakeWord model
                    prediction = owwModel.predict(audio_data)

                    # Column titles
                    n_spaces = 16
                    output_string_header = """
            Model Name         | Score | Wakeword Status
            --------------------------------------
            """

                    wake_word_detected = False
                    for mdl in owwModel.prediction_buffer.keys():
                        # Add scores in formatted table
                        scores = list(owwModel.prediction_buffer[mdl])
                        curr_score = format(scores[-1], ".20f").replace("-", "")

                        # Check for wake word detection
                        if scores[-1] > 0.5:
                            wake_word_detected = True
                            status_text = "Wakeword Detected!"
                        else:
                            status_text = "--" + " " * 20

                        output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {status_text}
            """

                    # Handle wake word detection with cooldown
                    current_time = time.time()
                    if (
                        wake_word_detected
                        and (current_time - last_detection_time) > detection_cooldown
                    ):
                        handle_wake_word_detection(args.light_ip, args.timeout)
                        last_detection_time = current_time

                    # Print results table
                    print("\033[F" * (4 * n_models + 1))
                    print(
                        output_string_header, "                             ", end="\r"
                    )
            except KeyboardInterrupt:
                print("\n🛑 Shutting down...")
            finally:
                if mic_stream:
                    mic_stream.close()
                if audio:
                    audio.terminate()

        except Exception as e:
            logger.error(f"Error in standalone mode: {e}")
            sys.exit(1)
    else:
        # Run FastAPI server
        logger.info("🚀 Starting Cerebus Wake Word & WebRTC Audio Server")
        uvicorn.run(app, host="0.0.0.0", port=8000)
