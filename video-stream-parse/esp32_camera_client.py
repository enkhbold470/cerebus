#!/usr/bin/env python3
"""
ESP32 Camera Stream Client
Captures video frames from ESP32 camera server for AI analysis
"""

import requests
import time
import io
from PIL import Image
import numpy as np
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ESP32CameraClient:
    def __init__(self, esp32_ip="192.168.18.39", port=80, stream_port=81):
        self.esp32_ip = esp32_ip
        self.port = port
        self.stream_port = stream_port
        self.base_url = f"http://{esp32_ip}:{port}"
        self.stream_url = "http://192.168.18.39:81/stream"  # Stream is working on port 81 (curl confirmed)
        self.capture_url = f"{self.base_url}/capture"
        self.control_url = f"{self.base_url}/control"
        self.status_url = f"{self.base_url}/status"
        
        # Headers to match the browser fetch request
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:137.0) Gecko/20100101 Firefox/137.0",
            "Accept": "image/avif,image/webp,image/png,image/svg+xml,image/*;q=0.8,*/*;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "en-US,en;q=0.5",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "close"  # Force close connections to avoid pooling issues
        }
        
        # Create session with custom adapter
        self.session = requests.Session()
        
        # Disable connection pooling and set aggressive timeout/retry settings
        adapter = HTTPAdapter(
            max_retries=Retry(total=1, backoff_factor=0.1),
            pool_connections=1,
            pool_maxsize=1
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def capture_single_frame(self):
        """Capture a single frame from the ESP32 camera"""
        try:
            response = self.session.get(
                self.capture_url, 
                headers=self.headers, 
                timeout=10,
                stream=False
            )
            response.raise_for_status()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(response.content))
            
            # Convert to numpy array for AI processing
            img_array = np.array(image)
            
            print(f"Captured frame: {img_array.shape}")
            return img_array
            
        except requests.exceptions.RequestException as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def set_camera_setting(self, variable, value):
        """Set camera parameters like quality, brightness, etc."""
        try:
            params = {'var': variable, 'val': value}
            response = self.session.get(
                self.control_url, 
                params=params, 
                headers=self.headers, 
                timeout=5
            )
            response.raise_for_status()
            print(f"Set {variable} to {value}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error setting {variable}: {e}")
            return False
    
    def get_camera_status(self):
        """Get current camera status and settings"""
        try:
            response = self.session.get(self.status_url, headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting status: {e}")
            return None
    
    def stream_frames_browser_style(self, max_frames=5):
        """
        Try to mimic exactly what the browser does
        """
        print("Attempting browser-style connection...")
        
        # Use a completely fresh session for each attempt
        fresh_session = requests.Session()
        
        # Minimal headers like browser
        browser_headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "*/*",
            "Connection": "keep-alive"
        }
        
        try:
            response = fresh_session.get(
                self.stream_url,
                headers=browser_headers,
                stream=True,
                timeout=30
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                print("✓ Connected to stream!")
                
                # Read first chunk to see if we get data
                chunk = next(response.iter_content(chunk_size=1024))
                print(f"First chunk size: {len(chunk)}")
                
                if b'Content-Type: image/jpeg' in chunk:
                    print("✓ Detected MJPEG stream!")
                    return True
                else:
                    print("Stream data:", chunk[:100])
            
            response.close()
            fresh_session.close()
            return False
            
        except Exception as e:
            print(f"Browser-style connection failed: {e}")
            fresh_session.close()
            return False


def main():
    # Configuration
    ESP32_IP = "192.168.18.39"  # ESP32 IP address from your fetch request
    
    # Initialize client
    client = ESP32CameraClient(ESP32_IP)
    
    print("ESP32 Camera Client")
    print("=" * 50)
    
    # Skip single frame test and go directly to streaming since web interface works
    print("Connecting directly to stream...")
    print(f"Stream URL: {client.stream_url}")
    
    try:
        # Test stream connection first
        import requests
        response = requests.get(client.stream_url, headers=client.headers, stream=True, timeout=5)
        if response.status_code == 200:
            print("✓ Stream connection successful!")
            response.close()
            
            # Start streaming frames
            print("\nStarting stream (Ctrl+C to stop)...")
            client.stream_frames_browser_style(max_frames=50)  # 50 frames with 1s delay for testing
        else:
            print(f"✗ Stream connection failed with status: {response.status_code}")
            
    except Exception as e:
        print(f"✗ Stream connection error: {e}")
        print("\nTrying alternative approaches...")
        
        # Try with simplified headers
        simple_headers = {"User-Agent": "Python ESP32 Client"}
        try:
            response = requests.get(client.stream_url, headers=simple_headers, stream=True, timeout=5)
            if response.status_code == 200:
                print("✓ Simple headers worked!")
                response.close()
                
                # Update client headers and try streaming
                client.headers = simple_headers
                client.stream_frames_browser_style(max_frames=20)
            else:
                print(f"✗ Simple headers also failed: {response.status_code}")
        except Exception as e2:
            print(f"✗ Simple headers error: {e2}")
            
            # Last resort: try without any custom headers
            try:
                response = requests.get(client.stream_url, stream=True, timeout=5)
                if response.status_code == 200:
                    print("✓ No headers worked!")
                    response.close()
                    
                    # Clear headers and try streaming
                    client.headers = {}
                    client.stream_frames_browser_style(max_frames=10)
                else:
                    print(f"✗ No headers failed: {response.status_code}")
            except Exception as e3:
                print(f"✗ All connection attempts failed: {e3}")
                print("Please check ESP32 IP address and network connectivity")


if __name__ == "__main__":
    main() 