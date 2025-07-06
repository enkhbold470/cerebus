import { useState, useCallback, useEffect } from 'react';
import mapsService from '../services/mapsService';
import type { 
  LocationCoordinates, 
  RouteRequest, 
  RouteResult, 
  NearbySearchRequest, 
  PlaceResult 
} from '../types/gps';

export interface UseMapsOptions {
  autoInitialize?: boolean;
  onError?: (error: string) => void;
}

export function useMaps(options: UseMapsOptions = {}) {
  const [isInitialized, setIsInitialized] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  const { autoInitialize = false, onError } = options;

  // Initialize Google Maps API
  const initialize = useCallback(async () => {
    if (isInitialized) return true;

    setIsLoading(true);
    setError(null);

    try {
      await mapsService.initialize();
      setIsInitialized(true);
      setIsLoading(false);
      return true;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to initialize Google Maps';
      setError(errorMsg);
      setIsLoading(false);
      onError?.(errorMsg);
      return false;
    }
  }, [isInitialized, onError]);

  // Auto-initialize if requested
  useEffect(() => {
    if (autoInitialize && !isInitialized && !isLoading) {
      initialize();
    }
  }, [autoInitialize, isInitialized, isLoading, initialize]);

  // Calculate route between two points
  const calculateRoute = useCallback(async (request: RouteRequest): Promise<RouteResult | null> => {
    if (!isInitialized) {
      const initialized = await initialize();
      if (!initialized) return null;
    }

    setIsLoading(true);
    setError(null);

    try {
      const route = await mapsService.calculateRoute(request);
      setIsLoading(false);
      return route;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to calculate route';
      setError(errorMsg);
      setIsLoading(false);
      onError?.(errorMsg);
      return null;
    }
  }, [isInitialized, initialize, onError]);

  // Search for nearby places
  const searchNearbyPlaces = useCallback(async (request: NearbySearchRequest): Promise<PlaceResult[]> => {
    if (!isInitialized) {
      const initialized = await initialize();
      if (!initialized) return [];
    }

    setIsLoading(true);
    setError(null);

    try {
      const places = await mapsService.searchNearbyPlaces(request);
      setIsLoading(false);
      return places;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to search places';
      setError(errorMsg);
      setIsLoading(false);
      onError?.(errorMsg);
      return [];
    }
  }, [isInitialized, initialize, onError]);

  // Geocode an address to coordinates
  const geocodeAddress = useCallback(async (address: string): Promise<LocationCoordinates | null> => {
    if (!isInitialized) {
      const initialized = await initialize();
      if (!initialized) return null;
    }

    setIsLoading(true);
    setError(null);

    try {
      const coordinates = await mapsService.geocodeAddress(address);
      setIsLoading(false);
      return coordinates;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to geocode address';
      setError(errorMsg);
      setIsLoading(false);
      onError?.(errorMsg);
      return null;
    }
  }, [isInitialized, initialize, onError]);

  // Reverse geocode coordinates to address
  const reverseGeocode = useCallback(async (location: LocationCoordinates): Promise<string | null> => {
    if (!isInitialized) {
      const initialized = await initialize();
      if (!initialized) return null;
    }

    setIsLoading(true);
    setError(null);

    try {
      const address = await mapsService.reverseGeocode(location);
      setIsLoading(false);
      return address;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to reverse geocode';
      setError(errorMsg);
      setIsLoading(false);
      onError?.(errorMsg);
      return null;
    }
  }, [isInitialized, initialize, onError]);

  // Calculate distance between two points
  const calculateDistance = useCallback(async (
    origin: LocationCoordinates, 
    destination: LocationCoordinates
  ): Promise<{
    distance: string;
    duration: string;
    distanceValue: number;
    durationValue: number;
  } | null> => {
    if (!isInitialized) {
      const initialized = await initialize();
      if (!initialized) return null;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await mapsService.calculateDistance(origin, destination);
      setIsLoading(false);
      return result;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to calculate distance';
      setError(errorMsg);
      setIsLoading(false);
      onError?.(errorMsg);
      return null;
    }
  }, [isInitialized, initialize, onError]);

  // Get current location using Google Maps API
  const getCurrentLocationWithMaps = useCallback(async (): Promise<LocationCoordinates | null> => {
    if (!isInitialized) {
      const initialized = await initialize();
      if (!initialized) return null;
    }

    setIsLoading(true);
    setError(null);

    try {
      const location = await mapsService.getCurrentLocationWithMaps();
      setIsLoading(false);
      return location;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to get current location';
      setError(errorMsg);
      setIsLoading(false);
      onError?.(errorMsg);
      return null;
    }
  }, [isInitialized, initialize, onError]);

  return {
    // State
    isInitialized,
    isLoading,
    error,
    
    // Methods
    initialize,
    calculateRoute,
    searchNearbyPlaces,
    geocodeAddress,
    reverseGeocode,
    calculateDistance,
    getCurrentLocationWithMaps,
    
    // Utilities
    isReady: () => mapsService.isReady(),
    getGoogleMapsAPI: () => mapsService.getGoogleMapsAPI(),
  };
} 