#!/usr/bin/env python3
"""
ESP32 MJPEG Stream Parser for macOS
Handles the continuous multipart stream for real-time AI processing
"""

import subprocess
import io
import time
import threading
import queue
from PIL import Image
import numpy as np
import signal
import sys

class ESP32StreamParser:
    def __init__(self, esp32_ip="192.168.18.39", stream_port=81):
        self.stream_url = f"http://{esp32_ip}:{stream_port}/stream"
        self.boundary = b"--123456789000000000000987654321"
        self.frame_queue = queue.Queue(maxsize=10)  # Buffer for frames
        self.is_running = False
        self.curl_process = None
        
    def start_stream(self):
        """Start the MJPEG stream parser in a separate thread"""
        self.is_running = True
        
        # Start curl process for streaming
        self.curl_process = subprocess.Popen(
            ["curl", "-s", "--no-buffer", self.stream_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Start parser thread
        parser_thread = threading.Thread(target=self._parse_stream, daemon=True)
        parser_thread.start()
        
        print(f"✓ Stream started from {self.stream_url}")
        return True
    
    def stop_stream(self):
        """Stop the stream"""
        self.is_running = False
        if self.curl_process:
            self.curl_process.terminate()
            self.curl_process.wait()
        print("✓ Stream stopped")
    
    def _parse_stream(self):
        """Parse the continuous MJPEG stream"""
        buffer = b""
        
        try:
            while self.is_running and self.curl_process:
                # Read data from curl
                chunk = self.curl_process.stdout.read(1024)
                if not chunk:
                    break
                
                buffer += chunk
                
                # Look for complete frames in buffer
                while True:
                    frame_data = self._extract_next_frame(buffer)
                    if frame_data is None:
                        break
                    
                    # Remove processed data from buffer
                    frame_end = buffer.find(self.boundary, len(frame_data))
                    if frame_end != -1:
                        buffer = buffer[frame_end:]
                    else:
                        break
                    
                    # Convert to numpy array
                    try:
                        image = Image.open(io.BytesIO(frame_data))
                        img_array = np.array(image)
                        
                        # Add to queue (non-blocking)
                        try:
                            self.frame_queue.put(img_array, block=False)
                        except queue.Full:
                            # Remove oldest frame if queue is full
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put(img_array, block=False)
                            except queue.Empty:
                                pass
                                
                    except Exception as e:
                        print(f"Frame conversion error: {e}")
                        
        except Exception as e:
            print(f"Stream parsing error: {e}")
    
    def _extract_next_frame(self, buffer):
        """Extract the next JPEG frame from buffer"""
        try:
            # Find boundary
            boundary_start = buffer.find(self.boundary)
            if boundary_start == -1:
                return None
            
            # Find next boundary
            next_boundary = buffer.find(self.boundary, boundary_start + len(self.boundary))
            if next_boundary == -1:
                return None
            
            # Extract frame section
            frame_section = buffer[boundary_start:next_boundary]
            
            # Find JPEG data (after headers)
            jpeg_start = frame_section.find(b'\xff\xd8')
            if jpeg_start == -1:
                return None
            
            jpeg_data = frame_section[jpeg_start:]
            
            # Find JPEG end
            jpeg_end = jpeg_data.find(b'\xff\xd9')
            if jpeg_end == -1:
                return None
            
            return jpeg_data[:jpeg_end + 2]
            
        except Exception:
            return None
    
    def get_latest_frame(self, timeout=1.0):
        """Get the most recent frame from the stream"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def process_frames(self, ai_callback=None, max_frames=None, fps_limit=5):
        """
        Process frames from the stream in real-time
        ai_callback: function to call with each frame for AI processing
        max_frames: maximum number of frames to process (None for unlimited)
        fps_limit: maximum frames per second to process
        """
        print(f"Processing stream frames (max_fps: {fps_limit})...")
        
        frame_count = 0
        last_time = time.time()
        min_interval = 1.0 / fps_limit
        
        try:
            while self.is_running:
                frame = self.get_latest_frame(timeout=2.0)
                
                if frame is not None:
                    # Rate limiting
                    current_time = time.time()
                    if current_time - last_time < min_interval:
                        continue
                    last_time = current_time
                    
                    frame_count += 1
                    print(f"Frame {frame_count}: {frame.shape}", end="")
                    
                    # Process with AI callback
                    if ai_callback:
                        try:
                            result = ai_callback(frame)
                            print(f" -> {result}")
                        except Exception as e:
                            print(f" -> AI Error: {e}")
                    else:
                        print(" -> Ready for AI")
                    
                    # Check if we've reached max frames
                    if max_frames and frame_count >= max_frames:
                        break
                else:
                    print("No frame received, stream may have ended")
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping stream processing...")
        
        self.stop_stream()
        print(f"Processed {frame_count} frames total")

def example_ai_processing(frame):
    """Example AI processing function"""
    height, width = frame.shape[:2]
    brightness = np.mean(frame)
    
    # Here you would run your actual AI model
    # result = your_model.predict(frame)
    
    return f"Size: {width}x{height}, Brightness: {brightness:.1f}"

def main():
    print("ESP32 MJPEG Stream Parser")
    print("=" * 40)
    
    parser = ESP32StreamParser()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nShutting down...")
        parser.stop_stream()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start streaming
    if parser.start_stream():
        print("Stream started successfully!")
        print("Press Ctrl+C to stop\n")
        
        # Process frames with AI
        parser.process_frames(
            ai_callback=example_ai_processing,
            max_frames=20,  # Process 20 frames then stop
            fps_limit=2     # Process at most 2 FPS
        )
    else:
        print("Failed to start stream")

if __name__ == "__main__":
    main() 