#!/usr/bin/env python3
"""
ESP32 AI Camera Client for macOS
Simple frame capture for AI processing using curl
"""

import subprocess
import io
import time
import json
import os
from PIL import Image
import numpy as np


class ESP32AIClient:
    def __init__(self, esp32_ip="192.168.18.39"):
        self.esp32_ip = esp32_ip
        self.capture_url = f"http://{esp32_ip}/capture"
        self.status_url = f"http://{esp32_ip}/status"
        self.control_url = f"http://{esp32_ip}/control"

    def capture_frame(self):
        """Capture a single frame for AI processing"""
        try:
            cmd = ["curl", "-s", "--max-time", "10", self.capture_url]
            result = subprocess.run(cmd, capture_output=True, timeout=15)

            if result.returncode == 0 and result.stdout:
                image = Image.open(io.BytesIO(result.stdout))
                img_array = np.array(image)
                return img_array
            else:
                return None

        except Exception as e:
            print(f"Capture error: {e}")
            return None

    def get_status(self):
        """Get camera status"""
        try:
            cmd = ["curl", "-s", "--max-time", "5", self.status_url]
            result = subprocess.run(cmd, capture_output=True, timeout=10)

            if result.returncode == 0 and result.stdout:
                return json.loads(result.stdout.decode())
            return None

        except Exception as e:
            return None

    def set_quality(self, quality):
        """Set JPEG quality (1-63, lower = better quality)"""
        return self._set_setting("quality", quality)

    def set_brightness(self, brightness):
        """Set brightness (-2 to 2)"""
        return self._set_setting("brightness", brightness)

    def set_framesize(self, framesize):
        """Set frame size (0-13, see ESP32 docs)"""
        return self._set_setting("framesize", framesize)

    def _set_setting(self, variable, value):
        """Set camera setting"""
        try:
            url = f"{self.control_url}?var={variable}&val={value}"
            cmd = ["curl", "-s", "--max-time", "5", url]
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return result.returncode == 0
        except:
            return False

    def capture_series(self, count=5, interval=2.0, save=True):
        """
        Capture a series of frames for AI processing
        Returns list of numpy arrays ready for AI models
        """
        print(f"Capturing {count} frames (interval: {interval}s)...")

        frames = []
        saved_paths = []

        if save:
            os.makedirs("ai_frames", exist_ok=True)

        for i in range(count):
            print(f"Frame {i+1}/{count}...", end=" ")

            frame = self.capture_frame()

            if frame is not None:
                frames.append(frame)
                print(f"✓ {frame.shape}")

                if save:
                    path = f"ai_frames/frame_{i+1:03d}.jpg"
                    Image.fromarray(frame).save(path)
                    saved_paths.append(path)

                # Here you would process with your AI model:
                # result = your_ai_model.predict(frame)
                # print(f"AI result: {result}")

            else:
                print("✗ Failed")

            if i < count - 1:
                time.sleep(interval)

        if save and saved_paths:
            print(f"\nSaved {len(saved_paths)} frames to ai_frames/")

        return frames


def example_ai_processing(frame):
    """
    Example of what you'd do with frames for AI
    Replace this with your actual AI model
    """
    # Example: Simple image analysis
    height, width = frame.shape[:2]
    avg_brightness = np.mean(frame)

    return {
        "size": f"{width}x{height}",
        "brightness": round(avg_brightness, 2),
        "total_pixels": width * height,
    }


def main():
    print("ESP32 AI Camera Client")
    print("=" * 40)

    client = ESP32AIClient()

    # Get camera info
    status = client.get_status()
    if status:
        print(f"Camera quality: {status.get('quality')}")
        print(f"Frame size: {status.get('framesize')}")
        print(f"Brightness: {status.get('brightness')}")

    # Configure camera for AI (optional)
    print("\nConfiguring camera...")
    client.set_quality(8)  # Good quality for AI
    client.set_brightness(0)  # Normal brightness

    # Test single frame
    print("\nTesting single frame...")
    frame = client.capture_frame()

    if frame is not None:
        print(f"✓ Frame captured: {frame.shape}")

        # Example AI processing
        result = example_ai_processing(frame)
        print(f"AI analysis: {result}")

        # Capture series for AI processing
        print("\nCapturing series for AI...")
        frames = client.capture_series(count=3, interval=2.0)

        print(f"\n✓ Ready for AI processing!")
        print(f"Captured {len(frames)} frames as numpy arrays")
        print("Each frame is ready for: model.predict(frame)")

    else:
        print("✗ Failed to capture frame")


if __name__ == "__main__":
    main()
