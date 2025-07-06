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

# Imports
import numpy as np
from openwakeword.model import Model
import argparse
import subprocess
import time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import io
import wave
import uvicorn
from datetime import datetime

app = FastAPI(title="Wake Word Detection Server", version="1.0.0")

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
    "--port",
    help="Port to run the server on",
    type=int,
    default=8000,
    required=False,
)
parser.add_argument(
    "--host",
    help="Host to run the server on",
    type=str,
    default="0.0.0.0",
    required=False,
)
parser.add_argument(
    "--save_audio",
    help="Save incoming audio files for debugging",
    action="store_true",
    required=False,
)
parser.add_argument(
    "--audio_dir",
    help="Directory to save audio files",
    type=str,
    default="./debug_audio",
    required=False,
)

args = parser.parse_args()

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

# Create audio debug directory if needed
if args.save_audio:
    debug_dir = Path(args.audio_dir)
    debug_dir.mkdir(exist_ok=True)
    print(f"💾 Audio files will be saved to: {debug_dir.absolute()}")

def save_audio_file(audio_data, filename_prefix, file_extension):
    """Save audio file for debugging"""
    if not args.save_audio:
        return None
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
        filename = f"{filename_prefix}_{timestamp}.{file_extension}"
        filepath = Path(args.audio_dir) / filename
        
        with open(filepath, "wb") as f:
            f.write(audio_data)
        
        print(f"💾 Saved audio file: {filepath}")
        return str(filepath)
    except Exception as e:
        print(f"⚠️ Failed to save audio file: {e}")
        return None

# Load pre-trained openwakeword models
if args.model_path != "":
    owwModel = Model(
        wakeword_models=[args.model_path], inference_framework=args.inference_framework
    )
else:
    owwModel = Model(inference_framework=args.inference_framework)


def convert_webm_to_wav(webm_data):
    """Convert webm audio data to wav format using ffmpeg"""
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_file:
            webm_file.write(webm_data)
            webm_path = webm_file.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name

        # Convert using ffmpeg
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                webm_path,
                "-ar",
                "16000",  # 16kHz sample rate
                "-ac",
                "1",  # Mono
                "-f",
                "wav",  # WAV format
                "-y",  # Overwrite output
                wav_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"❌ FFmpeg conversion failed: {result.stderr}")
            return None

        # Read the converted WAV file
        with open(wav_path, "rb") as f:
            wav_data = f.read()

        # Clean up temporary files
        os.unlink(webm_path)
        os.unlink(wav_path)

        return wav_data

    except Exception as e:
        print(f"❌ Error converting webm to wav: {e}")
        return None


def process_wav_data(wav_data):
    """Process WAV data and extract audio samples"""
    try:
        # Read WAV data
        wav_io = io.BytesIO(wav_data)
        with wave.open(wav_io, "rb") as wav_file:
            # Check format
            if wav_file.getnchannels() != 1:
                print(f"⚠️ Expected mono audio, got {wav_file.getnchannels()} channels")
            if wav_file.getframerate() != 16000:
                print(f"⚠️ Expected 16kHz, got {wav_file.getframerate()}Hz")
            if wav_file.getsampwidth() != 2:
                print(f"⚠️ Expected 16-bit audio, got {wav_file.getsampwidth() * 8}-bit")

            # Read audio data
            frames = wav_file.readframes(wav_file.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)

            return audio_data
    except Exception as e:
        print(f"❌ Error processing WAV data: {e}")
        return None


def detect_wake_word(audio_data):
    """Process audio data for wake word detection - matches microphone.py logic"""
    try:
        chunk_size = args.chunk_size
        detections = []
        
        print(f"🔍 Processing {len(audio_data)} samples in chunks of {chunk_size}")

        # Process audio in chunks like the microphone version
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]

            # Pad chunk if too short (same as microphone version)
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), "constant")

            # Feed to openWakeWord model (identical to microphone version)
            prediction = owwModel.predict(chunk)

            # Check for wake word detection (matches microphone.py logic)
            for mdl in owwModel.prediction_buffer.keys():
                scores = list(owwModel.prediction_buffer[mdl])
                curr_score = scores[-1]
                
                # Same threshold as microphone version
                if curr_score > 0.5:
                    detection = {
                        "model": mdl,
                        "score": float(curr_score),  # Convert numpy.float32 to Python float
                        "timestamp": time.time(),
                        "chunk_index": i // chunk_size,
                    }
                    detections.append(detection)
                    
                    # Debug output matching microphone version
                    print(f"🔥 WAKE WORD DETECTED!")
                    print(f"   Model: {mdl}")
                    print(f"   Score: {curr_score:.6f}")
                    print(f"   Chunk: {i // chunk_size}")
                    print(f"   Time: {time.strftime('%H:%M:%S')}")

        return detections

    except Exception as e:
        print(f"❌ Error in wake word detection: {e}")
        return []


@app.post("/detect")
async def detect_endpoint(audio: UploadFile = File(...)):
    """Endpoint to receive audio data and detect wake words"""
    try:
        print(f"📨 Received POST request at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Get audio data from request
        if not audio:
            raise HTTPException(status_code=400, detail="No audio file provided")

        audio_data = await audio.read()

        print(f"📁 Received audio file: {len(audio_data)} bytes")

        # Save original audio file for debugging
        original_filename = None
        if audio_data.startswith(b"RIFF") and b"WAVE" in audio_data[:12]:
            original_filename = save_audio_file(audio_data, "original", "wav")
        else:
            original_filename = save_audio_file(audio_data, "original", "webm")

        # Try to process as WAV first, then convert from webm if needed
        wav_data = None

        # Check if it's already WAV format
        if audio_data.startswith(b"RIFF") and b"WAVE" in audio_data[:12]:
            print("🎵 Detected WAV format")
            wav_data = audio_data
        else:
            print(f"🎵 Detected other format, converting with ffmpeg... (first 20 bytes: {audio_data[:20]})")
            wav_data = convert_webm_to_wav(audio_data)
            
            # Save converted WAV file for debugging
            if wav_data is not None:
                save_audio_file(wav_data, "converted", "wav")

        # Handle conversion failures gracefully
        if wav_data is None:
            print("⚠️ Audio conversion failed - unsupported format or corrupted file")
            return {
                "error": "unsupported_audio_format",
                "message": "Could not process audio file. Please ensure it's a valid WebM or WAV file.",
                "detections": [],
                "audio_length": 0,
                "sample_rate": 16000
            }

        # Extract audio samples
        audio_samples = process_wav_data(wav_data)
        if audio_samples is None:
            print("⚠️ Audio samples extraction failed - invalid WAV data")
            return {
                "error": "invalid_wav_data", 
                "message": "Could not extract audio samples from WAV data.",
                "detections": [],
                "audio_length": 0,
                "sample_rate": 16000
            }

        print(f"🔊 Processed {len(audio_samples)} audio samples")

        # Detect wake words
        detections = detect_wake_word(audio_samples)

        # Print final results
        if detections:
            print(f"💡 [DEBUG] Light would be triggered here! ({len(detections)} detections)")
        else:
            print("🔍 No wake word detected")

        return {
            "detections": detections,
            "audio_length": len(audio_samples),
            "sample_rate": 16000,
        }

    except Exception as e:
        print(f"❌ Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": args.model_path,
        "framework": args.inference_framework,
        "models_loaded": len(owwModel.models.keys()),
    }


if __name__ == "__main__":
    print("\n\n")
    print("#" * 100)
    print("🚀 Wake Word Detection Server Starting...")
    print(f"Model: {args.model_path}")
    print(f"Framework: {args.inference_framework}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Endpoints:")
    print(f"  POST /detect - Send audio for wake word detection")
    print(f"  GET  /health - Health check")
    print("#" * 100)
    print("")

    # Start the FastAPI server with uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
