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
import pyaudio
import numpy as np
from openwakeword.model import Model
import argparse
import subprocess
import threading
import time
from pathlib import Path

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
            ["curl", "-s", "-m", "10", url],
            capture_output=True,
            text=True,
            timeout=15
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
            ["curl", "-s", "-m", "10", url],
            capture_output=True,
            text=True,
            timeout=15
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

    # Schedule light off after timeout (placeholder for agentic features)
    def delayed_light_off():
        print(
            f"⏱️ Waiting {timeout_seconds} seconds (placeholder for agentic processing)..."
        )
        time.sleep(timeout_seconds)
        turn_off_light(ip_address)
        print("💡 Light control sequence completed\n")

    # Run light off in separate thread to avoid blocking detection
    threading.Thread(target=delayed_light_off, daemon=True).start()


# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size
audio = pyaudio.PyAudio()
mic_stream = audio.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)

# Load pre-trained openwakeword models
if args.model_path != "":
    owwModel = Model(
        wakeword_models=[args.model_path], inference_framework=args.inference_framework
    )
else:
    owwModel = Model(inference_framework=args.inference_framework)

n_models = len(owwModel.models.keys())

# Run capture loop continuosly, checking for wakewords
if __name__ == "__main__":
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

    while True:
        # Get audio
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Feed to openWakeWord model
        prediction = owwModel.predict(audio)

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
        print(output_string_header, "                             ", end="\r")
