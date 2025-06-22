import type { LocationData, LocationCoordinates, GPSSession, ServerDataPayload } from '../types/gps';

class GPSService {
  private watchId: number | null = null;
  private currentSession: GPSSession | null = null;
  private onLocationUpdate?: (location: LocationData) => void;
  private onError?: (error: string) => void;

  // Get current location (one-time)
  async getCurrentLocation(): Promise<LocationData> {
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject(new Error('Geolocation is not supported by this browser'));
        return;
      }

      navigator.geolocation.getCurrentPosition(
        (position) => {
          const locationData: LocationData = {
            lat: position.coords.latitude,
            lng: position.coords.longitude,
            accuracy: position.coords.accuracy,
            timestamp: Date.now(),
            speed: position.coords.speed || undefined,
            heading: position.coords.heading || undefined,
          };
          resolve(locationData);
        },
        (error) => {
          reject(new Error(`GPS Error: ${error.message}`));
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 60000,
        }
      );
    });
  }

  // Start continuous GPS tracking
  startTracking(
    onLocationUpdate: (location: LocationData) => void,
    onError?: (error: string) => void,
    sessionId?: string
  ): string {
    if (!navigator.geolocation) {
      onError?.('Geolocation is not supported by this browser');
      throw new Error('Geolocation not supported');
    }

    // Stop any existing tracking
    this.stopTracking();

    const id = sessionId || `gps_${Date.now()}`;
    
    this.currentSession = {
      sessionId: id,
      startTime: Date.now(),
      locationHistory: [],
      isTracking: true,
    };

    this.onLocationUpdate = onLocationUpdate;
    this.onError = onError;

    this.watchId = navigator.geolocation.watchPosition(
      (position) => {
        const locationData: LocationData = {
          lat: position.coords.latitude,
          lng: position.coords.longitude,
          accuracy: position.coords.accuracy,
          timestamp: Date.now(),
          speed: position.coords.speed || undefined,
          heading: position.coords.heading || undefined,
        };

        // Update session
        if (this.currentSession) {
          this.currentSession.currentLocation = locationData;
          this.currentSession.locationHistory.push(locationData);
          
          // Keep only last 100 locations to manage memory
          if (this.currentSession.locationHistory.length > 100) {
            this.currentSession.locationHistory = this.currentSession.locationHistory.slice(-100);
          }
        }

        // Call the callback
        this.onLocationUpdate?.(locationData);

        // Push to server
        this.pushToServer({
          type: 'gps_update',
          timestamp: Date.now(),
          sessionId: id,
          data: locationData,
        });
      },
      (error) => {
        const errorMessage = `GPS Error: ${error.message}`;
        this.onError?.(errorMessage);
      },
      {
        enableHighAccuracy: true,
        timeout: 5000,
        maximumAge: 30000,
      }
    );

    return id;
  }

  // Stop GPS tracking
  stopTracking(): void {
    if (this.watchId !== null) {
      navigator.geolocation.clearWatch(this.watchId);
      this.watchId = null;
    }

    if (this.currentSession) {
      this.currentSession.isTracking = false;
    }

    this.onLocationUpdate = undefined;
    this.onError = undefined;
  }

  // Get current session data
  getCurrentSession(): GPSSession | null {
    return this.currentSession;
  }

  // Calculate distance between two points (Haversine formula)
  static calculateDistance(point1: LocationCoordinates, point2: LocationCoordinates): number {
    const R = 6371e3; // Earth's radius in meters
    const lat1 = (point1.lat * Math.PI) / 180;
    const lat2 = (point2.lat * Math.PI) / 180;
    const deltaLat = ((point2.lat - point1.lat) * Math.PI) / 180;
    const deltaLng = ((point2.lng - point1.lng) * Math.PI) / 180;

    const a =
      Math.sin(deltaLat / 2) * Math.sin(deltaLat / 2) +
      Math.cos(lat1) * Math.cos(lat2) * Math.sin(deltaLng / 2) * Math.sin(deltaLng / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return R * c;
  }

  // Check if geolocation is available
  static isGeolocationSupported(): boolean {
    return 'geolocation' in navigator;
  }

  // Check current permission status
  static async checkPermission(): Promise<PermissionState> {
    if (!navigator.permissions) {
      return 'prompt';
    }
    
    try {
      const permission = await navigator.permissions.query({ name: 'geolocation' });
      return permission.state;
    } catch {
      return 'prompt';
    }
  }

  // Request geolocation permission
  static async requestPermission(): Promise<boolean> {
    try {
      await new Promise<GeolocationPosition>((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject, {
          timeout: 1000,
          maximumAge: Infinity,
        });
      });
      return true;
    } catch {
      return false;
    }
  }

  // Push data to server
  private async pushToServer(payload: ServerDataPayload): Promise<void> {
    console.log('📍 GPS Data for Server:', {
      ...payload,
      formattedData: {
        sessionId: payload.sessionId,
        type: payload.type,
        timestamp: new Date(payload.timestamp).toISOString(),
        location: payload.data,
      }
    });

    try {
      const response = await fetch('http://localhost:8000/gps/location', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sessionId: payload.sessionId,
          ...payload.data as LocationData,
        }),
      });

      if (!response.ok) {
        console.warn('Failed to push GPS data to server:', response.statusText);
      } else {
        const result = await response.json();
        console.log('✅ GPS data pushed to server:', result.message);
      }
    } catch (error) {
      console.error('❌ Error pushing GPS data to server:', error);
    }
  }

  // Get location history summary
  getLocationHistory(): LocationData[] {
    return this.currentSession?.locationHistory || [];
  }

  // Get tracking statistics
  getTrackingStats() {
    if (!this.currentSession) return null;

    const history = this.currentSession.locationHistory;
    if (history.length < 2) return null;

    let totalDistance = 0;
    let maxSpeed = 0;
    
    for (let i = 1; i < history.length; i++) {
      const distance = GPSService.calculateDistance(history[i - 1], history[i]);
      totalDistance += distance;
      
      if (history[i].speed && history[i].speed! > maxSpeed) {
        maxSpeed = history[i].speed!;
      }
    }

    const duration = Date.now() - this.currentSession.startTime;

    return {
      sessionId: this.currentSession.sessionId,
      duration: duration,
      totalDistance: totalDistance,
      averageSpeed: duration > 0 ? (totalDistance / (duration / 1000)) : 0,
      maxSpeed: maxSpeed,
      pointsRecorded: history.length,
      isActive: this.currentSession.isTracking,
    };
  }
}

// Export singleton instance
export const gpsService = new GPSService();
export { GPSService };
export default gpsService; 