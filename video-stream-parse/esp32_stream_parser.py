#!/usr/bin/env python3
"""
ESP32 MJPEG Stream Parser for macOS
Handles the continuous multipart stream for real-time AI processing with YOLO object detection
"""

import subprocess
import io
import time
import threading
import queue
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import signal
import sys
import cv2

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️  YOLO not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False


class ESP32StreamParser:
    def __init__(
        self,
        esp32_ip="192.168.18.39",
        stream_port=81,
        save_images=False,
        output_dir="captured_frames",
        image_format="jpg",
        save_fps_limit=1,
        flip_180=True,
        enable_yolo=True,
        yolo_model="yolov8n.pt",
        yolo_confidence=0.5,
        save_annotated=True,
    ):
        self.stream_url = f"http://{esp32_ip}:{stream_port}/stream"
        self.boundary = b"--123456789000000000000987654321"
        self.frame_queue = queue.Queue(maxsize=10)  # Buffer for frames
        self.save_queue = queue.Queue(maxsize=50)  # Buffer for saving images
        self.is_running = False
        self.curl_process = None

        # Image saving configuration
        self.save_images = save_images
        self.output_dir = output_dir
        self.image_format = image_format.lower()
        self.save_fps_limit = save_fps_limit
        self.saved_frame_count = 0
        self.last_save_time = 0
        self.flip_180 = flip_180
        self.save_annotated = save_annotated

        # YOLO configuration
        self.enable_yolo = enable_yolo and YOLO_AVAILABLE
        self.yolo_model_path = yolo_model
        self.yolo_confidence = yolo_confidence
        self.yolo_model = None
        self.detection_count = 0

        # Initialize YOLO model
        if self.enable_yolo:
            self._initialize_yolo()

        # Create output directory if saving is enabled
        if self.save_images:
            self._setup_output_directory()

    def _initialize_yolo(self):
        """Initialize YOLO model"""
        try:
            print(f"🔄 Loading YOLO model: {self.yolo_model_path}")
            self.yolo_model = YOLO(self.yolo_model_path)
            print(f"✓ YOLO model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load YOLO model: {e}")
            self.enable_yolo = False

    def _setup_output_directory(self):
        """Create output directory with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{self.output_dir}_{timestamp}"

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"✓ Created output directory: {self.output_dir}")
            
            # Create subdirectories for different types of images
            if self.save_annotated and self.enable_yolo:
                os.makedirs(os.path.join(self.output_dir, "annotated"), exist_ok=True)
                os.makedirs(os.path.join(self.output_dir, "original"), exist_ok=True)
                
        except Exception as e:
            print(f"✗ Failed to create output directory: {e}")
            self.save_images = False

    def start_stream(self):
        """Start the MJPEG stream parser in a separate thread"""
        self.is_running = True

        # Start curl process for streaming
        self.curl_process = subprocess.Popen(
            ["curl", "-s", "--no-buffer", self.stream_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Start parser thread
        parser_thread = threading.Thread(target=self._parse_stream, daemon=True)
        parser_thread.start()

        # Start image saver thread if enabled
        if self.save_images:
            saver_thread = threading.Thread(
                target=self._save_images_worker, daemon=True
            )
            saver_thread.start()

        print(f"✓ Stream started from {self.stream_url}")
        if self.save_images:
            print(f"✓ Image saving enabled - output: {self.output_dir}")
        if self.flip_180:
            print("✓ 180° rotation enabled")
        if self.enable_yolo:
            print(f"✓ YOLO object detection enabled (confidence: {self.yolo_confidence})")
        return True

    def stop_stream(self):
        """Stop the stream"""
        self.is_running = False
        if self.curl_process:
            self.curl_process.terminate()
            self.curl_process.wait()

        # Wait a moment for save queue to finish
        if self.save_images:
            time.sleep(1)
            print(f"✓ Saved {self.saved_frame_count} images to {self.output_dir}")
        
        if self.enable_yolo:
            print(f"✓ Processed {self.detection_count} YOLO detections")

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

                        # Flip image 180 degrees if enabled
                        if self.flip_180:
                            image = image.rotate(180)

                        img_array = np.array(image)

                        # Add to processing queue (non-blocking)
                        try:
                            self.frame_queue.put((img_array, image.copy()), block=False)
                        except queue.Full:
                            # Remove oldest frame if queue is full
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put((img_array, image.copy()), block=False)
                            except queue.Empty:
                                pass

                        # Add to save queue if enabled (rate limited)
                        if self.save_images:
                            current_time = time.time()
                            save_interval = 1.0 / self.save_fps_limit

                            if current_time - self.last_save_time >= save_interval:
                                try:
                                    # Store the PIL image for saving (already rotated if needed)
                                    self.save_queue.put(image.copy(), block=False)
                                    self.last_save_time = current_time
                                except queue.Full:
                                    # Skip saving this frame if queue is full
                                    pass

                    except Exception as e:
                        print(f"Frame conversion error: {e}")

        except Exception as e:
            print(f"Stream parsing error: {e}")

    def _save_images_worker(self):
        """Background worker to save images asynchronously"""
        while self.is_running:
            try:
                # Get image from save queue (already rotated if needed)
                image = self.save_queue.get(timeout=1.0)

                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
                    :-3
                ]  # milliseconds
                filename = f"frame_{timestamp}.{self.image_format}"
                
                # Save original image
                if self.save_annotated and self.enable_yolo:
                    filepath = os.path.join(self.output_dir, "original", filename)
                else:
                    filepath = os.path.join(self.output_dir, filename)

                # Save image
                if self.image_format in ["jpg", "jpeg"]:
                    image.save(filepath, "JPEG", quality=85)
                elif self.image_format == "png":
                    image.save(filepath, "PNG")
                else:
                    image.save(filepath)

                self.saved_frame_count += 1

                if self.saved_frame_count % 10 == 0:
                    print(f"💾 Saved {self.saved_frame_count} images")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Image save error: {e}")

    def detect_objects_yolo(self, image_array):
        """Perform YOLO object detection on image array"""
        if not self.enable_yolo or self.yolo_model is None:
            return None, []

        try:
            # Run YOLO detection
            results = self.yolo_model(image_array, conf=self.yolo_confidence, verbose=False)
            
            detections = []
            annotated_image = None
            
            if len(results) > 0:
                result = results[0]
                
                # Get annotated image
                annotated_array = result.plot()
                annotated_image = Image.fromarray(cv2.cvtColor(annotated_array, cv2.COLOR_BGR2RGB))
                
                # Extract detection info
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]
                        
                        detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name
                        }
                        detections.append(detection)
                
                self.detection_count += 1
            
            return annotated_image, detections

        except Exception as e:
            print(f"YOLO detection error: {e}")
            return None, []

    def save_annotated_image(self, annotated_image):
        """Save annotated image with detections"""
        if not self.save_images or not self.save_annotated or annotated_image is None:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"annotated_{timestamp}.{self.image_format}"
            filepath = os.path.join(self.output_dir, "annotated", filename)
            
            if self.image_format in ["jpg", "jpeg"]:
                annotated_image.save(filepath, "JPEG", quality=85)
            elif self.image_format == "png":
                annotated_image.save(filepath, "PNG")
            else:
                annotated_image.save(filepath)
                
        except Exception as e:
            print(f"Annotated image save error: {e}")

    def _extract_next_frame(self, buffer):
        """Extract the next JPEG frame from buffer"""
        try:
            # Find boundary
            boundary_start = buffer.find(self.boundary)
            if boundary_start == -1:
                return None

            # Find next boundary
            next_boundary = buffer.find(
                self.boundary, boundary_start + len(self.boundary)
            )
            if next_boundary == -1:
                return None

            # Extract frame section
            frame_section = buffer[boundary_start:next_boundary]

            # Find JPEG data (after headers)
            jpeg_start = frame_section.find(b"\xff\xd8")
            if jpeg_start == -1:
                return None

            jpeg_data = frame_section[jpeg_start:]

            # Find JPEG end
            jpeg_end = jpeg_data.find(b"\xff\xd9")
            if jpeg_end == -1:
                return None

            return jpeg_data[: jpeg_end + 2]

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
        Process frames from the stream in real-time with YOLO object detection
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
                frame_data = self.get_latest_frame(timeout=2.0)

                if frame_data is not None:
                    img_array, pil_image = frame_data
                    
                    # Rate limiting
                    current_time = time.time()
                    if current_time - last_time < min_interval:
                        continue
                    last_time = current_time

                    frame_count += 1
                    status_msg = f"Frame {frame_count}: {img_array.shape}"

                    if self.save_images:
                        status_msg += f" (saved: {self.saved_frame_count})"

                    # Perform YOLO detection
                    if self.enable_yolo:
                        annotated_image, detections = self.detect_objects_yolo(img_array)
                        
                        if detections:
                            status_msg += f" -> {len(detections)} objects: "
                            status_msg += ", ".join([f"{det['class_name']}({det['confidence']:.2f})" for det in detections[:3]])
                            if len(detections) > 3:
                                status_msg += f" +{len(detections)-3} more"
                                
                            # Save annotated image
                            self.save_annotated_image(annotated_image)
                        else:
                            status_msg += " -> No objects detected"
                    
                    print(status_msg)

                    # Process with additional AI callback
                    if ai_callback:
                        try:
                            if self.enable_yolo and 'detections' in locals():
                                result = ai_callback(img_array, detections)
                            else:
                                result = ai_callback(img_array)
                            print(f"   AI Result: {result}")
                        except Exception as e:
                            print(f"   AI Error: {e}")

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


def yolo_ai_processing(frame, detections=None):
    """Enhanced AI processing function with YOLO detections"""
    height, width = frame.shape[:2]
    brightness = np.mean(frame)
    
    result = f"Size: {width}x{height}, Brightness: {brightness:.1f}"
    
    if detections:
        result += f", Objects: {len(detections)}"
        # Analyze detected objects
        object_types = {}
        for det in detections:
            obj_type = det['class_name']
            if obj_type in object_types:
                object_types[obj_type] += 1
            else:
                object_types[obj_type] = 1
        
        result += f" ({dict(object_types)})"
    
    return result


def main():
    print("ESP32 MJPEG Stream Parser with YOLO Object Detection")
    print("=" * 60)

    # Enable YOLO object detection with configuration
    parser = ESP32StreamParser(
        save_images=True,           # Enable image saving
        output_dir="yolo_detections", # Base directory name
        image_format="jpg",         # Save as JPEG
        save_fps_limit=0.5,         # Save 1 image every 2 seconds
        flip_180=True,              # Flip images 180 degrees
        enable_yolo=True,           # Enable YOLO detection
        yolo_model="yolov8n.pt",    # YOLO model (nano for speed)
        yolo_confidence=0.5,        # Detection confidence threshold
        save_annotated=True,        # Save images with bounding boxes
    )

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

        # Process frames with YOLO detection
        parser.process_frames(
            ai_callback=yolo_ai_processing,
            max_frames=100,  # Process 100 frames then stop
            fps_limit=2      # Process at most 2 FPS
        )
    else:
        print("Failed to start stream")


if __name__ == "__main__":
    main()
