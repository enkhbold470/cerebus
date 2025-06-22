#!/usr/bin/env python3
"""
Simple Navigation API Server
HTTP server for image-based navigation guidance
"""

import json
import time
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse as urlparse
from simple_navigation import SimpleNavigation
from PIL import Image
import tempfile
import os

class NavigationAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for navigation API"""
    
    def __init__(self, *args, **kwargs):
        # Initialize navigation system
        self.nav_system = SimpleNavigation()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse.urlparse(self.path)
        path = parsed_path.path
        
        if path == '/':
            self.send_api_info()
        elif path == '/status':
            self.send_status()
        elif path == '/test':
            self.send_test_response()
        else:
            self.send_error(404, "Endpoint not found")
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse.urlparse(self.path)
        path = parsed_path.path
        
        if path == '/analyze':
            self.analyze_image()
        else:
            self.send_error(404, "Endpoint not found")
    
    def send_api_info(self):
        """Send API documentation"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            'name': 'Simple Navigation API',
            'version': '1.0.0',
            'description': 'Provides navigation guidance for visually impaired users',
            'endpoints': {
                '/': 'GET - API documentation',
                '/status': 'GET - System status',
                '/analyze': 'POST - Analyze image for navigation (send image as base64 or multipart)',
                '/test': 'GET - Test with sample data'
            },
            'usage': {
                'analyze_endpoint': {
                    'method': 'POST',
                    'content_type': 'application/json',
                    'body': {
                        'image_base64': 'base64 encoded image data',
                        'image_format': 'jpg or png'
                    }
                }
            }
        }
        
        self.wfile.write(json.dumps(response, indent=2).encode())
    
    def send_status(self):
        """Send system status"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            'status': 'active',
            'navigation_system': 'ready',
            'yolo_model': 'yolov8n.pt',
            'supported_formats': ['jpg', 'jpeg', 'png'],
            'timestamp': time.time()
        }
        
        self.wfile.write(json.dumps(response, indent=2).encode())
    
    def analyze_image(self):
        """Analyze uploaded image for navigation"""
        try:
            # Get content length
            content_length = int(self.headers['Content-Length'])
            
            # Read POST data
            post_data = self.rfile.read(content_length)
            
            # Parse JSON data
            try:
                data = json.loads(post_data.decode('utf-8'))
            except json.JSONDecodeError:
                self.send_error_response(400, "Invalid JSON data")
                return
            
            # Check if image data is provided
            if 'image_base64' not in data:
                self.send_error_response(400, "No image_base64 field provided")
                return
            
            # Decode base64 image
            try:
                image_data = base64.b64decode(data['image_base64'])
            except Exception as e:
                self.send_error_response(400, f"Invalid base64 data: {str(e)}")
                return
            
            # Save image to temporary file
            temp_file = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_file.write(image_data)
                    temp_file_path = temp_file.name
                
                # Analyze image
                result = self.nav_system.analyze_image(temp_file_path)
                
                # Send successful response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                result['timestamp'] = time.time()
                self.wfile.write(json.dumps(result, indent=2).encode())
                
            finally:
                # Clean up temporary file
                if temp_file and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            
        except Exception as e:
            self.send_error_response(500, f"Analysis failed: {str(e)}")
    
    def send_test_response(self):
        """Send test response with sample navigation data"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        # Sample navigation result
        sample_result = {
            'status': 'success',
            'direction': 'slight_left',
            'instruction': 'Move slightly to the left',
            'confidence': 0.75,
            'detected_objects': [
                {
                    'object': 'person',
                    'confidence': 0.85,
                    'position': (320, 240),
                    'angle': 25.0,
                    'distance': 0.4,
                    'priority': 7,
                    'is_obstacle': True
                },
                {
                    'object': 'chair',
                    'confidence': 0.72,
                    'position': (450, 200),
                    'angle': 45.0,
                    'distance': 0.6,
                    'priority': 5,
                    'is_obstacle': True
                }
            ],
            'safe_zones': [
                {'zone': 'far_left', 'safety_score': 10.0, 'recommended': True},
                {'zone': 'left', 'safety_score': 8.5, 'recommended': True},
                {'zone': 'slight_left', 'safety_score': 7.2, 'recommended': True},
                {'zone': 'center', 'safety_score': 3.2, 'recommended': False},
                {'zone': 'slight_right', 'safety_score': 4.8, 'recommended': False},
                {'zone': 'right', 'safety_score': 6.1, 'recommended': False},
                {'zone': 'far_right', 'safety_score': 9.0, 'recommended': True}
            ],
            'warnings': ['Obstacle directly ahead: person'],
            'timestamp': time.time(),
            'test_mode': True
        }
        
        self.wfile.write(json.dumps(sample_result, indent=2).encode())
    
    def send_error_response(self, code, message):
        """Send error response"""
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        error_response = {
            'status': 'error',
            'error_code': code,
            'message': message,
            'timestamp': time.time()
        }
        
        self.wfile.write(json.dumps(error_response, indent=2).encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Override to customize logging"""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {format % args}")

def run_server(port=5001):
    """Run the navigation API server"""
    server_address = ('0.0.0.0', port)
    
    # Create server with navigation system
    class NavigationServer(HTTPServer):
        def __init__(self, server_address, RequestHandlerClass):
            super().__init__(server_address, RequestHandlerClass)
            self.nav_system = SimpleNavigation()
    
    httpd = NavigationServer(server_address, NavigationAPIHandler)
    
    print("Simple Navigation API Server")
    print("=" * 50)
    print(f"Server URL: http://localhost:{port}")
    print(f"Status: http://localhost:{port}/status")
    print(f"Test: http://localhost:{port}/test")
    print("\nTo analyze image:")
    print("POST http://localhost:{port}/analyze")
    print("Body: {\"image_base64\": \"<base64_encoded_image>\"}")
    print("\nPress Ctrl+C to stop")
    print("=" * 50)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()

if __name__ == '__main__':
    run_server()
