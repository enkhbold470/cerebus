#!/usr/bin/env python3
"""
ESP32 Camera Stream Client for macOS
Uses curl subprocess to bypass macOS HTTP security restrictions
"""

import subprocess
import io
import time
import json
from PIL import Image
import numpy as np
import tempfile
import os

class ESP32StreamMacOS:
    def __init__(self, esp32_ip="192.168.18.39", port=80, stream_port=81):
        self.esp32_ip = esp32_ip
        self.port = port  
        self.stream_port = stream_port
        self.stream_url = f"http://{esp32_ip}:{stream_port}/stream"
        self.capture_url = f"http://{esp32_ip}:{port}/capture"
        self.status_url = f"http://{esp32_ip}:{port}/status"
        self.control_url = f"http://{esp32_ip}:{port}/control"
    
    def capture_single_frame(self):
        """Capture a single frame using curl to bypass macOS security"""
        try:
            # Use curl to capture a single frame
            cmd = [
                'curl', '-s', '--max-time', '10', 
                '--connect-timeout', '5',
                self.capture_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=15)
            
            if result.returncode == 0 and result.stdout:
                # Convert to PIL Image
                image = Image.open(io.BytesIO(result.stdout))
                img_array = np.array(image)
                print(f"✓ Captured frame: {img_array.shape}")
                return img_array
            else:
                print(f"✗ Curl capture failed (return code: {result.returncode})")
                return None
                
        except Exception as e:
            print(f"✗ Frame capture error: {e}")
            return None
    
    def get_camera_status(self):
        """Get camera status using curl"""
        try:
            cmd = ['curl', '-s', '--max-time', '5', self.status_url]
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            
            if result.returncode == 0 and result.stdout:
                return json.loads(result.stdout.decode())
            else:
                print(f"✗ Status request failed (return code: {result.returncode})")
                return None
                
        except Exception as e:
            print(f"✗ Status error: {e}")
            return None
    
    def set_camera_setting(self, variable, value):
        """Set camera setting using curl"""
        try:
            url = f"{self.control_url}?var={variable}&val={value}"
            cmd = ['curl', '-s', '--max-time', '5', url]
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            
            if result.returncode == 0:
                print(f"✓ Set {variable} = {value}")
                return True
            else:
                print(f"✗ Failed to set {variable} (return code: {result.returncode})")
                return False
                
        except Exception as e:
            print(f"✗ Setting error: {e}")
            return False
    
    def stream_frames_to_files(self, num_frames=5, delay=2.0, save_dir="frames"):
        """
        Capture frames from stream and save to files for AI processing
        This works around macOS restrictions by saving frames locally
        """
        print(f"Capturing {num_frames} frames from stream...")
        
        # Create directory for frames
        os.makedirs(save_dir, exist_ok=True)
        
        frames = []
        
        for i in range(num_frames):
            print(f"\nCapturing frame {i+1}/{num_frames}...")
            
            # Create temporary file for this frame
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Use curl to capture stream data for a short time and save first frame
                cmd = [
                    'curl', '-s', '--max-time', '5',
                    '--output', tmp_path,
                    self.stream_url
                ]
                
                # Run curl in background and kill after short time to get one frame
                process = subprocess.Popen(cmd)
                time.sleep(1.5)  # Let it capture some data
                process.terminate()
                process.wait()
                
                # Try to extract frame from captured data
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    frame = self._extract_frame_from_stream_file(tmp_path)
                    
                    if frame is not None:
                        # Save frame
                        frame_path = os.path.join(save_dir, f"frame_{i+1:03d}.jpg")
                        Image.fromarray(frame).save(frame_path)
                        frames.append(frame)
                        
                        print(f"✓ Saved frame {i+1}: {frame.shape} -> {frame_path}")
                        
                        # Here you would process with your AI model
                        # result = your_ai_analysis(frame)
                        
                    else:
                        print(f"✗ Could not extract frame {i+1}")
                else:
                    print(f"✗ No data captured for frame {i+1}")
                
            except Exception as e:
                print(f"✗ Error capturing frame {i+1}: {e}")
            
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
            # Wait before next frame
            if i < num_frames - 1:
                time.sleep(delay)
        
        print(f"\n✓ Captured {len(frames)} frames successfully")
        print(f"Frames saved in: {save_dir}/")
        return frames
    
    def _extract_frame_from_stream_file(self, filepath):
        """Extract first JPEG frame from stream data file"""
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # Find JPEG markers
            jpeg_start = data.find(b'\xff\xd8')
            if jpeg_start == -1:
                return None
            
            jpeg_end = data.find(b'\xff\xd9', jpeg_start)
            if jpeg_end == -1:
                return None
            
            # Extract JPEG
            jpeg_data = data[jpeg_start:jpeg_end + 2]
            
            # Convert to numpy array
            image = Image.open(io.BytesIO(jpeg_data))
            return np.array(image)
            
        except Exception as e:
            print(f"Frame extraction error: {e}")
            return None

def main():
    print("ESP32 Camera Client for macOS")
    print("=" * 50)
    
    client = ESP32StreamMacOS()
    
    # Test single frame capture first
    print("Testing single frame capture...")
    frame = client.capture_single_frame()
    
    if frame is not None:
        print("✓ Single frame capture works!")
        
        # Get camera status
        status = client.get_camera_status()
        if status:
            print(f"Camera quality: {status.get('quality', 'unknown')}")
            print(f"Frame size: {status.get('framesize', 'unknown')}")
        
        # Capture stream frames
        print("\nCapturing frames from stream...")
        frames = client.stream_frames_to_files(num_frames=3, delay=3.0)
        
        if frames:
            print(f"\n✓ Successfully captured {len(frames)} frames!")
            print("Frames are ready for AI processing")
            print("Check the 'frames/' directory for saved images")
        
    else:
        print("✗ Frame capture failed")
        print("Make sure ESP32 is accessible at http://192.168.18.39")

if __name__ == "__main__":
    main() 