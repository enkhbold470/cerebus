import { useState, useEffect, useCallback, useRef } from 'react';
import gpsService, { GPSService } from '../services/gpsService';
import type { LocationData, GeolocationResult, GPSSession } from '../types/gps';

export interface UseGeolocationOptions {
  enableHighAccuracy?: boolean;
  timeout?: number;
  maximumAge?: number;
  watchPosition?: boolean;
  onLocationUpdate?: (location: LocationData) => void;
  onError?: (error: string) => void;
}

export interface TrackingStats {
  sessionId: string;
  duration: number;
  totalDistance: number;
  averageSpeed: number;
  maxSpeed: number;
  pointsRecorded: number;
  isActive: boolean;
}

export function useGeolocation(options: UseGeolocationOptions = {}) {
  const [location, setLocation] = useState<LocationData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isTracking, setIsTracking] = useState<boolean>(false);
  const [permission, setPermission] = useState<PermissionState>('prompt');
  
  const sessionIdRef = useRef<string | null>(null);
  const {
    watchPosition = false,
    onLocationUpdate,
    onError,
  } = options;

  // Check initial permission state
  useEffect(() => {
    const checkPermission = async () => {
      try {
        const permissionState = await GPSService.checkPermission();
        setPermission(permissionState);
      } catch {
        console.warn('Could not check geolocation permission');
      }
    };

    if (GPSService.isGeolocationSupported()) {
      checkPermission();
    } else {
      setError('Geolocation is not supported by this browser');
    }
  }, []);

  // Get current location (one-time)
  const getCurrentLocation = useCallback(async () => {
    if (!GPSService.isGeolocationSupported()) {
      const errorMsg = 'Geolocation is not supported by this browser';
      setError(errorMsg);
      onError?.(errorMsg);
      return null;
    }

    setIsLoading(true);
    setError(null);

    try {
      const locationData = await gpsService.getCurrentLocation();
      setLocation(locationData);
      onLocationUpdate?.(locationData);
      setPermission('granted');
      return locationData;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to get location';
      setError(errorMsg);
      onError?.(errorMsg);
      
      if (errorMsg.includes('denied')) {
        setPermission('denied');
      }
      
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [onLocationUpdate, onError]);

  // Start location tracking
  const startTracking = useCallback(() => {
    if (!GPSService.isGeolocationSupported()) {
      const errorMsg = 'Geolocation is not supported by this browser';
      setError(errorMsg);
      onError?.(errorMsg);
      return;
    }

    if (isTracking) {
      return; // Already tracking
    }

    setIsLoading(true);
    setError(null);

    try {
      const sessionId = gpsService.startTracking(
        (locationData) => {
          setLocation(locationData);
          setIsLoading(false);
          onLocationUpdate?.(locationData);
          setPermission('granted');
        },
        (errorMsg) => {
          setError(errorMsg);
          setIsLoading(false);
          onError?.(errorMsg);
          
          if (errorMsg.includes('denied')) {
            setPermission('denied');
          }
        }
      );
      
      sessionIdRef.current = sessionId;
      setIsTracking(true);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to start tracking';
      setError(errorMsg);
      setIsLoading(false);
      onError?.(errorMsg);
    }
  }, [isTracking, onLocationUpdate, onError]);

  // Stop location tracking
  const stopTracking = useCallback(() => {
    gpsService.stopTracking();
    setIsTracking(false);
    sessionIdRef.current = null;
  }, []);

  // Request permission
  const requestPermission = useCallback(async () => {
    try {
      const granted = await GPSService.requestPermission();
      setPermission(granted ? 'granted' : 'denied');
      return granted;
    } catch {
      setPermission('denied');
      return false;
    }
  }, []);

  // Auto-start tracking if watchPosition is enabled
  useEffect(() => {
    if (watchPosition && permission === 'granted' && !isTracking) {
      startTracking();
    }

    return () => {
      if (watchPosition) {
        stopTracking();
      }
    };
  }, [watchPosition, permission, isTracking, startTracking, stopTracking]);

  // Get location history
  const getLocationHistory = useCallback(() => {
    return gpsService.getLocationHistory();
  }, []);

  // Get tracking statistics
  const getTrackingStats = useCallback((): TrackingStats | null => {
    return gpsService.getTrackingStats();
  }, []);

  // Get current session
  const getCurrentSession = useCallback((): GPSSession | null => {
    return gpsService.getCurrentSession();
  }, []);

  const result: GeolocationResult & {
    isTracking: boolean;
    permission: PermissionState;
    sessionId: string | null;
    getCurrentLocation: () => Promise<LocationData | null>;
    startTracking: () => void;
    stopTracking: () => void;
    requestPermission: () => Promise<boolean>;
    getLocationHistory: () => LocationData[];
    getTrackingStats: () => TrackingStats | null;
    getCurrentSession: () => GPSSession | null;
  } = {
    location: location || { lat: 0, lng: 0, timestamp: 0 },
    error: error || undefined,
    isLoading,
    isTracking,
    permission,
    sessionId: sessionIdRef.current,
    getCurrentLocation,
    startTracking,
    stopTracking,
    requestPermission,
    getLocationHistory,
    getTrackingStats,
    getCurrentSession,
  };

  return result;
} 