# GPS & Maps Integration Setup

This directory contains the GPS and mapping functionality extracted from `gps-visual` and integrated into the cerebus-web React application.

## 🏗️ Architecture

```
src/
├── types/gps.ts              # TypeScript interfaces for GPS/Maps data
├── services/
│   ├── gpsService.ts         # Browser geolocation service
│   └── mapsService.ts        # Google Maps API integration
├── hooks/
│   ├── useGeolocation.ts     # React hook for GPS functionality
│   └── useMaps.ts            # React hook for Maps functionality
└── components/
    └── GPSMapsDemo.tsx       # Demo component showcasing functionality
```

## 🚀 Features

### GPS Service (`gpsService.ts`)
- ✅ Get current location (one-time)
- ✅ Continuous location tracking
- ✅ Permission management
- ✅ Location history and statistics
- ✅ Distance calculations using Haversine formula
- ✅ All GPS data logged to console for server integration

### Maps Service (`mapsService.ts`)
- ✅ Google Maps API integration
- ✅ Route calculation with turn-by-turn directions
- ✅ Place search (nearby businesses, landmarks)
- ✅ Geocoding (address → coordinates)
- ✅ Reverse geocoding (coordinates → address)
- ✅ Distance matrix calculations
- ✅ All Maps data logged to console for server integration

### React Hooks
- ✅ `useGeolocation()` - Complete GPS functionality with React state
- ✅ `useMaps()` - Google Maps API with React integration
- ✅ Error handling and loading states
- ✅ Permission management

## ⚙️ Configuration

### 1. Google Maps API Key

Create a `.env.local` file in the `cerebus-web` directory:

```bash
# Get your API key from: https://console.cloud.google.com/apis/credentials
VITE_GOOGLE_MAPS_API_KEY=your_api_key_here
```

### 2. Required Google Cloud APIs

Enable these APIs in your Google Cloud Console:
- **Maps JavaScript API** - For map rendering
- **Places API** - For place searches
- **Directions API** - For route calculations  
- **Geocoding API** - For address conversion

### 3. API Key Restrictions (Recommended)

In Google Cloud Console, restrict your API key:
- **Application restrictions**: HTTP referrers
- **API restrictions**: Only the APIs listed above
- **Referrer restrictions**: Your domain(s)

## 📱 Usage Examples

### Basic GPS Usage

```typescript
import { useGeolocation } from '../hooks/useGeolocation';

function MyComponent() {
  const gps = useGeolocation({
    watchPosition: true, // Auto-start tracking
    onLocationUpdate: (location) => {
      console.log('New location:', location);
    }
  });

  return (
    <div>
      {gps.isLoading && <p>Getting location...</p>}
      {gps.location && (
        <p>
          Location: {gps.location.lat}, {gps.location.lng}
          (±{gps.location.accuracy}m)
        </p>
      )}
      <button onClick={() => gps.startTracking()}>
        Start Tracking
      </button>
    </div>
  );
}
```

### Basic Maps Usage

```typescript
import { useMaps } from '../hooks/useMaps';

function RouteComponent() {
  const maps = useMaps({ autoInitialize: true });
  const [route, setRoute] = useState(null);

  const calculateRoute = async () => {
    const result = await maps.calculateRoute({
      origin: { lat: 40.7128, lng: -74.0060 }, // NYC
      destination: { lat: 40.7589, lng: -73.9851 }, // Times Square
      travelMode: 'WALKING'
    });
    setRoute(result);
  };

  return (
    <div>
      <button onClick={calculateRoute}>Calculate Route</button>
      {route && (
        <div>
          <p>Distance: {route.route_summary.distance}</p>
          <p>Duration: {route.route_summary.duration}</p>
        </div>
      )}
    </div>
  );
}
```

## 🔗 Server Integration

All GPS updates, route calculations, and place searches are automatically logged to the browser console in a structured format ready for server integration:

```javascript
// GPS Updates
console.log('📍 GPS Data for Server:', {
  type: 'gps_update',
  sessionId: 'gps_1640995200000',
  timestamp: 1640995200000,
  data: {
    lat: 40.7128,
    lng: -74.0060,
    accuracy: 10,
    timestamp: 1640995200000
  }
});

// Route Calculations
console.log('🗺️ Maps Data for Server:', {
  type: 'route_calculated',
  sessionId: 'route_1640995200000',
  timestamp: 1640995200000,
  data: {
    // Complete RouteResult object
  }
});
```

To integrate with your backend, replace the `console.log` calls in the services with actual POST requests:

```typescript
// In gpsService.ts or mapsService.ts
private async pushToServer(payload: ServerDataPayload): Promise<void> {
  try {
    await fetch('/api/gps-data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  } catch (error) {
    console.error('Failed to push data to server:', error);
  }
}
```

## 🎮 Demo Component

The `GPSMapsDemo` component provides a complete testing interface:

1. **Permission Management** - Request location access
2. **GPS Testing** - Get location, start/stop tracking
3. **Route Calculation** - Enter destination, get turn-by-turn directions
4. **Place Search** - Find nearby businesses/landmarks
5. **Address Lookup** - Get current address from coordinates
6. **Real-time Logging** - See all data structures in the activity log

## 🔧 Integration with App.tsx

Add the demo component to your main app:

```typescript
import { GPSMapsDemo } from './components/GPSMapsDemo';

function App() {
  return (
    <div>
      {/* Your existing app content */}
      <GPSMapsDemo />
    </div>
  );
}
```

## 📊 Data Structures

All data follows the TypeScript interfaces defined in `types/gps.ts`:

- `LocationData` - GPS coordinates with metadata
- `RouteResult` - Complete route with turn-by-turn instructions
- `PlaceResult` - Business/landmark information
- `ServerDataPayload` - Standardized format for server communication

## 🛡️ Privacy & Permissions

- Location permission is requested explicitly
- Users can grant/deny location access
- Tracking can be started/stopped at any time
- Location history is limited to last 100 points
- All data processing happens client-side

## 🐛 Troubleshooting

### Common Issues

1. **"Google Maps API key not found"**
   - Add `VITE_GOOGLE_MAPS_API_KEY` to `.env.local`
   - Restart the development server

2. **"Geolocation not supported"**
   - Ensure HTTPS in production
   - Check browser compatibility

3. **"API key restrictions"**
   - Verify API key has correct APIs enabled
   - Check referrer restrictions

4. **"Location permission denied"**
   - Guide users to enable location in browser settings
   - Provide fallback manual location entry

### Development Notes

- GPS tracking works best with HTTPS
- Google Maps requires valid API key even in development
- Console logs show detailed error information
- Use browser dev tools to simulate different locations

## 🚀 Next Steps

For full server integration:

1. Replace console.log calls with API requests
2. Implement server endpoints to receive GPS/Maps data
3. Add database storage for location history
4. Implement real-time updates via WebSockets
5. Add map visualization components
6. Integrate with the Cerebus navigation system 