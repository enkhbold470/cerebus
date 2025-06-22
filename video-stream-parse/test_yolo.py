#!/usr/bin/env python3
"""
Test YOLO object detection functionality
"""

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import cv2
import os
import glob


def test_yolo_detection():
    """Test YOLO detection with a sample image"""
    print("🔄 Testing YOLO Object Detection")
    print("=" * 40)

    # Load YOLO model
    try:
        model = YOLO("yolov8n.pt")  # This will download the model if not present
        print("✓ YOLO model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        return False

    # Create a test image with some geometric shapes
    print("🔄 Creating test image...")
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    test_image.fill(50)  # Dark gray background

    # Add some colored rectangles to simulate objects
    cv2.rectangle(test_image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue square
    cv2.rectangle(
        test_image, (300, 150), (500, 300), (0, 255, 0), -1
    )  # Green rectangle
    cv2.circle(test_image, (400, 450), 80, (0, 0, 255), -1)  # Red circle

    # Add some text
    cv2.putText(
        test_image, "TEST", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3
    )

    print("✓ Test image created")

    # Run YOLO detection
    print("🔄 Running YOLO detection...")
    try:
        results = model(test_image, conf=0.25, verbose=True)

        if len(results) > 0:
            result = results[0]

            # Get annotated image
            annotated_array = result.plot()
            annotated_image = Image.fromarray(
                cv2.cvtColor(annotated_array, cv2.COLOR_BGR2RGB)
            )

            # Save the test images
            test_pil = Image.fromarray(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
            test_pil.save("test_original.jpg", quality=85)
            annotated_image.save("test_annotated.jpg", quality=85)

            print("✓ Test images saved: test_original.jpg, test_annotated.jpg")

            # Extract detection info
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Get confidence and class
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]

                    detection = {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(confidence),
                        "class_id": class_id,
                        "class_name": class_name,
                    }
                    detections.append(detection)

                    print(
                        f"  📍 Detected: {class_name} (confidence: {confidence:.2f}) at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]"
                    )

            print(f"✓ YOLO detection completed: {len(detections)} objects found")

        else:
            print("ℹ️  No objects detected in test image")

        return True

    except Exception as e:
        print(f"✗ YOLO detection failed: {e}")
        return False


def test_yolo_on_videos():
    """Test YOLO detection on actual test videos"""
    print("\n🔄 Testing YOLO on Test Videos")
    print("=" * 40)

    # Load YOLO model
    try:
        model = YOLO("yolov8n.pt")
        print("✓ YOLO model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        return False

    # Find video files in test_videos directory
    video_dir = "../test_videos"
    if not os.path.exists(video_dir):
        print(f"✗ Test videos directory not found: {video_dir}")
        return False

    video_files = glob.glob(os.path.join(video_dir, "*.MOV")) + glob.glob(
        os.path.join(video_dir, "*.mov")
    )

    if not video_files:
        print(f"✗ No video files found in {video_dir}")
        return False

    print(f"📹 Found {len(video_files)} video files:")
    for video_file in video_files:
        print(f"  - {os.path.basename(video_file)}")

    # Process each video
    all_detections = {}

    for video_file in video_files:
        video_name = os.path.basename(video_file)
        print(f"\n🔄 Processing {video_name}...")
        print("🎮 Controls: SPACE = pause/resume, 'q' = next video, ESC = exit all")

        try:
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print(f"✗ Failed to open video: {video_name}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            print(
                f"  📊 Video info: {frame_count} frames, {fps:.1f} FPS, {duration:.1f}s duration"
            )

            # Create window
            window_name = f"YOLO Detection - {video_name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1200, 800)

            video_detections = []
            frame_num = 0
            paused = False
            exit_all = False

            # Calculate delay for real-time playback
            delay = int(1000 / fps) if fps > 0 else 33  # milliseconds

            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"  ✓ Finished playing {video_name}")
                        break

                    # Run YOLO detection
                    results = model(frame, conf=0.3, verbose=False)

                    # Get annotated frame
                    if len(results) > 0:
                        annotated_frame = results[0].plot()

                        # Extract detection info
                        frame_detections = []
                        if results[0].boxes is not None:
                            for box in results[0].boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                confidence = box.conf[0].cpu().numpy()
                                class_id = int(box.cls[0].cpu().numpy())
                                class_name = model.names[class_id]

                                detection = {
                                    "frame": frame_num,
                                    "time": frame_num / fps if fps > 0 else 0,
                                    "bbox": [
                                        float(x1),
                                        float(y1),
                                        float(x2),
                                        float(y2),
                                    ],
                                    "confidence": float(confidence),
                                    "class_name": class_name,
                                }
                                frame_detections.append(detection)

                        if frame_detections:
                            video_detections.extend(frame_detections)

                        # Add frame info overlay
                        info_text = f"Frame: {frame_num}/{frame_count} | Time: {frame_num/fps:.1f}s | Objects: {len(frame_detections)}"
                        cv2.putText(
                            annotated_frame,
                            info_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )

                        # Add detection count by class
                        if frame_detections:
                            class_counts = {}
                            for det in frame_detections:
                                class_name = det["class_name"]
                                class_counts[class_name] = (
                                    class_counts.get(class_name, 0) + 1
                                )

                            y_offset = 60
                            for class_name, count in class_counts.items():
                                text = f"{class_name}: {count}"
                                cv2.putText(
                                    annotated_frame,
                                    text,
                                    (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 255, 255),
                                    2,
                                )
                                y_offset += 25

                        # Add pause indicator
                        if paused:
                            cv2.putText(
                                annotated_frame,
                                "PAUSED - Press SPACE to resume",
                                (10, annotated_frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2,
                            )
                    else:
                        annotated_frame = frame.copy()
                        info_text = f"Frame: {frame_num}/{frame_count} | Time: {frame_num/fps:.1f}s | No detections"
                        cv2.putText(
                            annotated_frame,
                            info_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )

                    # Display frame
                    cv2.imshow(window_name, annotated_frame)
                    frame_num += 1

                # Handle key presses
                key = cv2.waitKey(delay if not paused else 0) & 0xFF

                if key == 27:  # ESC key - exit all videos
                    exit_all = True
                    break
                elif key == ord("q"):  # 'q' key - next video
                    break
                elif key == ord(" "):  # SPACE key - pause/resume
                    paused = not paused
                    print(f"  {'⏸️  Paused' if paused else '▶️  Resumed'}")

            cap.release()
            cv2.destroyWindow(window_name)

            if exit_all:
                print("  🚪 Exiting all videos...")
                break

            # Summarize detections for this video
            if video_detections:
                all_detections[video_name] = video_detections

                # Count objects by class
                class_counts = {}
                for detection in video_detections:
                    class_name = detection["class_name"]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                print(f"  ✓ {video_name}: {len(video_detections)} total detections")
                print(f"    📊 Object summary:")
                for class_name, count in sorted(class_counts.items()):
                    print(f"      - {class_name}: {count} detections")

                # Save detection data to file
                detection_file = f"{video_name}_detections.txt"
                with open(detection_file, "w") as f:
                    f.write(f"Detections for {video_name}\n")
                    f.write("=" * 50 + "\n\n")
                    for detection in video_detections:
                        f.write(
                            f"Frame {detection['frame']} ({detection['time']:.1f}s): "
                            f"{detection['class_name']} (conf: {detection['confidence']:.2f}) "
                            f"at {detection['bbox']}\n"
                        )
                print(f"    💾 Saved detection log: {detection_file}")
            else:
                print(f"  ℹ️  {video_name}: No objects detected")

        except Exception as e:
            print(f"✗ Error processing {video_name}: {e}")
            cv2.destroyAllWindows()
            continue

    cv2.destroyAllWindows()

    # Overall summary
    print(f"\n📊 OVERALL SUMMARY:")
    print(f"  🎬 Videos processed: {len(video_files)}")
    print(f"  📍 Videos with detections: {len(all_detections)}")

    total_detections = sum(len(detections) for detections in all_detections.values())
    print(f"  🎯 Total detections: {total_detections}")

    if all_detections:
        # Combined class summary
        all_class_counts = {}
        for video_detections in all_detections.values():
            for detection in video_detections:
                class_name = detection["class_name"]
                all_class_counts[class_name] = all_class_counts.get(class_name, 0) + 1

        print(f"  📋 All detected objects:")
        for class_name, count in sorted(all_class_counts.items()):
            print(f"    - {class_name}: {count} detections")

    return len(all_detections) > 0


def test_stream_parser_yolo():
    """Test the ESP32StreamParser with YOLO (without actual stream)"""
    print("\n🔄 Testing ESP32StreamParser YOLO Integration")
    print("=" * 50)

    from esp32_stream_parser import ESP32StreamParser

    # Create parser with YOLO enabled (but don't start stream)
    parser = ESP32StreamParser(
        enable_yolo=True,
        yolo_model="yolov8n.pt",
        yolo_confidence=0.5,
    )

    if parser.enable_yolo:
        print("✓ ESP32StreamParser YOLO integration working")

        # Test detection on sample image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        annotated_image, detections = parser.detect_objects_yolo(test_image)

        print(f"✓ Detection method working: {len(detections)} objects detected")
        return True
    else:
        print("✗ ESP32StreamParser YOLO integration failed")
        return False


if __name__ == "__main__":
    print("YOLO Object Detection Test Suite")
    print("=" * 60)

    # Test 1: Basic YOLO functionality
    test1_passed = test_yolo_detection()

    # Test 2: YOLO on actual test videos
    test2_passed = test_yolo_on_videos()

    # Test 3: ESP32StreamParser integration
    test3_passed = test_stream_parser_yolo()

    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    print(f"  Basic YOLO Detection: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"  Video Processing: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print(f"  StreamParser Integration: {'✓ PASSED' if test3_passed else '✗ FAILED'}")

    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 All tests passed! YOLO integration is ready.")
        print("💡 You can now run the ESP32 stream with YOLO detection:")
        print("   python esp32_stream_parser.py")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
