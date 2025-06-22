# Smart Glasses ReAct AI Agent

This is the AI Agent component of the smart glasses navigation system, built using Google's Agent Development Kit (ADK). It serves as the central decision engine that coordinates between image analysis, navigation, and audio response systems.

## Architecture Overview

Based on the system diagram, this agent implements the "ReAct AI Agent" component that:

1. **Receives input** from:
   - Image Analysis Tools (Gemini Vision, YOLO, SegNet)
   - Navigation Tools (GPS Handler)
   - Wake Word Detection triggers

2. **Makes decisions** using ReAct methodology:
   - **Reason**: Analyze current situation and context
   - **Act**: Take appropriate navigation action
   - **Observe**: Process feedback and results

3. **Provides output** through:
   - Audio Response Tools (navigation instructions to AirPods)

## Multi-Agent Architecture

The system uses specialized sub-agents following the Google ADK pattern:

### 1. Image Analysis Agent
- **Purpose**: Process visual data from ESP32 camera
- **Tools**: 
  - Gemini Vision API integration
  - YOLO object detection
  - SegNet road segmentation
- **Output**: Scene understanding, object detection, path analysis

### 2. Navigation Agent  
- **Purpose**: Handle GPS data and route planning
- **Tools**:
  - GPS coordinate processing
  - Route instruction generation
  - Location context via Google Search
- **Output**: Turn-by-turn directions, location awareness

### 3. Audio Response Agent
- **Purpose**: Generate appropriate audio feedback
- **Tools**:
  - Navigation instruction formatting
  - Safety alert generation
- **Output**: Concise, actionable audio messages

### 4. ReAct Decision Engine (Root Agent)
- **Purpose**: Coordinate all sub-agents and make final decisions
- **Methodology**: Reason → Act → Observe cycle
- **Priority Framework**: Safety → Navigation → Context

## Setup Instructions

### 1. Environment Configuration

1. Copy `env_template.txt` to `.env`
2. Configure your settings:

```bash
# Required: Get your Gemini API key from Google AI Studio
GOOGLE_API_KEY=your_actual_api_key_here

# Optional: For Vertex AI (set to TRUE if using)
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1

# System endpoints (update with your actual URLs)
ESP32_CAMERA_URL=http://192.168.1.100/stream
PHONE_GPS_ENDPOINT=http://localhost:8080/gps
```

### 2. Install Dependencies

```bash
cd agent
pip install -r requirements.txt
```

### 3. Integration Points

The agent includes placeholder functions that need to be connected to your existing systems:

#### Vision Processing Pipeline
- `analyze_with_gemini_vision()` → Connect to your Gemini Vision API
- `detect_objects_yolo()` → Connect to your YOLO model
- `segment_image_segnet()` → Connect to your SegNet model

#### Hardware Interfaces  
- `process_gps_data()` → Connect to your GPS Handler
- `generate_navigation_audio()` → Connect to your Audio Response Tools

### 4. Run the Agent

#### Using Poetry Scripts (Recommended)
```bash
# Setup environment (first time only)
poetry run agent-setup

# Run tests to verify setup
poetry run agent-test

# Start web interface
poetry run agent-web

# Start CLI interface  
poetry run agent-cli

# Debug agent configuration
poetry run agent-debug
```

#### Direct ADK Commands
```bash
# Terminal interface
adk run .

# Web UI interface  
adk web
```

#### Integration with Your System
```python
from agent.agent import root_agent

# Example usage in your processing server
async def handle_wake_word_trigger(visual_data, gps_data, user_query):
    context = {
        "visual_scene": visual_data,
        "gps_coordinates": gps_data, 
        "user_input": user_query
    }
    
    response = await root_agent.process(context)
    return response
```

## Testing the Agent

### Sample Interactions

1. **Basic Navigation Query**:
   ```
   User: "Where should I go?"
   Expected: Agent analyzes GPS, provides direction guidance
   ```

2. **Safety Scenario**:
   ```
   User: "What's ahead of me?"
   Expected: Agent analyzes visual scene, warns of obstacles
   ```

3. **Complex Navigation**:
   ```
   User: "Guide me to the nearest coffee shop"
   Expected: Agent coordinates GPS + search + route planning
   ```

## Integration with Existing Components

This agent is designed to integrate with your existing system components:

```
ESP32 Camera → Video Handler → Vision Pipeline → Image Analysis Agent
AirPods Audio → Audio Handler → Wake Word → ReAct Decision Engine  
Phone GPS → GPS Handler → Navigation Tools → Navigation Agent
```

The agent receives processed data from your handlers and returns structured responses for your audio output system.

## Development Notes

### Extending the Agent

1. **Add new tools**: Create functions and add to appropriate agent's `tools` list
2. **Modify behavior**: Update the `instruction` field for each agent
3. **Add new agents**: Create specialized agents for specific functions

### Performance Considerations

- The agent prioritizes safety-critical responses
- Audio responses are optimized for real-time delivery (<10 seconds)
- Uses Gemini 2.5 Flash for speed and efficiency
- Supports streaming responses for better user experience

### Monitoring and Debugging

The ADK provides built-in debugging through:
- Web UI for interactive testing
- Detailed execution logs
- Step-by-step agent decision tracking

## Next Steps

1. **Connect Integration Points**: Wire up the placeholder functions to your actual systems
2. **Test End-to-End**: Verify the complete pipeline from hardware input to audio output  
3. **Tune Instructions**: Refine agent prompts based on real-world testing
4. **Add Memory**: Implement conversation memory for multi-turn interactions
5. **Deploy**: Use ADK's deployment tools for production

## Troubleshooting

### Common Issues

1. **Agent not responding**: Check your `GOOGLE_API_KEY` in `.env`
2. **Tool errors**: Verify integration functions are properly connected
3. **Slow responses**: Consider using faster model variants or optimizing prompts

### Getting Help

- [ADK Documentation](https://github.com/google/adk-samples)
- [Google AI Studio](https://aistudio.google.com) for API keys
- Review the system logs in ADK web UI for detailed error information 