import { Loader } from '@googlemaps/js-api-loader';
import type { 
  LocationCoordinates, 
  RouteRequest, 
  RouteResult, 
  RouteStep,
  NearbySearchRequest, 
  PlaceResult,
  ServerDataPayload 
} from '../types/gps';

class MapsService {
  private loader: Loader | null = null;
  private isLoaded: boolean = false;
  private loadPromise: Promise<typeof google> | null = null;

  constructor() {
    // Get API key from environment variables
    const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
    
    if (!apiKey) {
      console.warn('Google Maps API key not found. Set VITE_GOOGLE_MAPS_API_KEY in your environment.');
      return;
    }

    this.loader = new Loader({
      apiKey: apiKey,
      version: 'weekly',
      libraries: ['places', 'geometry'],
    });
  }

  // Initialize the Google Maps API
  async initialize(): Promise<typeof google> {
    if (!this.loader) {
      throw new Error('Google Maps API key not configured');
    }

    if (this.isLoaded) {
      return window.google;
    }

    if (this.loadPromise) {
      return this.loadPromise;
    }

    this.loadPromise = this.loader.load().then((google) => {
      this.isLoaded = true;
      console.log('Google Maps API loaded successfully');
      return google;
    });

    return this.loadPromise;
  }

  // Calculate route between two points
  async calculateRoute(request: RouteRequest): Promise<RouteResult> {
    const google = await this.initialize();
    
    const directionsService = new google.maps.DirectionsService();
    
    // Convert our travel mode to Google Maps format
    const travelMode = this.getTravelMode(request.travelMode || 'WALKING');
    
    return new Promise((resolve, reject) => {
      directionsService.route(
        {
          origin: new google.maps.LatLng(request.origin.lat, request.origin.lng),
          destination: new google.maps.LatLng(request.destination.lat, request.destination.lng),
          travelMode: travelMode,
          unitSystem: google.maps.UnitSystem.METRIC,
          avoidHighways: false,
          avoidTolls: false,
        },
        (result, status) => {
          if (status === google.maps.DirectionsStatus.OK && result) {
            const route = result.routes[0];
            const leg = route.legs[0];

            const startInstruction: RouteStep = {
              instruction: `Starting navigation to ${request.destinationName || 'your destination'}. Begin ${request.travelMode?.toLowerCase() || 'walking'}.`,
              distance: '',
              duration: '',
              isStart: true,
            };

            const endInstruction: RouteStep = {
              instruction: `You have arrived at ${request.destinationName || 'your destination'}`,
              distance: '',
              duration: '',
              isEnd: true,
            };

            const steps: RouteStep[] = leg.steps.map((step) => ({
              instruction: step.instructions,
              distance: step.distance?.text || '',
              duration: step.duration?.text || '',
              originalInstruction: step.instructions,
              isStep: true,
            }));

            const routeResult: RouteResult = {
              timestamp: new Date().toISOString(),
              origin: {
                lat: leg.start_location.lat(),
                lng: leg.start_location.lng(),
                address: leg.start_address,
              },
              destination: {
                lat: leg.end_location.lat(),
                lng: leg.end_location.lng(),
                address: leg.end_address,
              },
              route_summary: {
                distance: leg.distance?.text || '',
                duration: leg.duration?.text || '',
                travel_mode: request.travelMode || 'WALKING',
              },
              voice_instructions: [startInstruction, ...steps, endInstruction],
              detailed_steps: leg.steps.map((step) => ({
                instruction: step.instructions,
                distance: step.distance?.text || '',
                duration: step.duration?.text || '',
                start_location: {
                  lat: step.start_location.lat(),
                  lng: step.start_location.lng(),
                },
                end_location: {
                  lat: step.end_location.lat(),
                  lng: step.end_location.lng(),
                },
              })),
              polyline: route.overview_polyline,
            };

            // Push route data to server
            this.pushToServer({
              type: 'route_calculated',
              timestamp: Date.now(),
              sessionId: 'default',
              data: routeResult,
            });

            resolve(routeResult);
          } else {
            reject(new Error(`Failed to calculate route: ${status}`));
          }
        }
      );
    });
  }

  // Search for nearby places
  async searchNearbyPlaces(request: NearbySearchRequest): Promise<PlaceResult[]> {
    const google = await this.initialize();
    
    const service = new google.maps.places.PlacesService(
      document.createElement('div')
    );

    return new Promise((resolve, reject) => {
      const searchRequest: google.maps.places.TextSearchRequest = {
        query: request.query,
        location: new google.maps.LatLng(request.location.lat, request.location.lng),
        radius: request.radius || 5000,
      };

      service.textSearch(searchRequest, (results, status) => {
        if (status === google.maps.places.PlacesServiceStatus.OK && results) {
          const places: PlaceResult[] = results.map((place) => ({
            place_id: place.place_id || '',
            name: place.name || 'Unknown',
            address: place.formatted_address || '',
            location: {
              lat: place.geometry?.location?.lat() || 0,
              lng: place.geometry?.location?.lng() || 0,
            },
            rating: place.rating,
            user_ratings_total: place.user_ratings_total,
            types: place.types,
          }));

          // Push places search data to server
          this.pushToServer({
            type: 'places_search',
            timestamp: Date.now(),
            sessionId: 'default',
            data: places,
          });

          resolve(places);
        } else {
          reject(new Error(`Places search failed: ${status}`));
        }
      });
    });
  }

  // Geocode an address to coordinates
  async geocodeAddress(address: string): Promise<LocationCoordinates> {
    const google = await this.initialize();
    
    const geocoder = new google.maps.Geocoder();

    return new Promise((resolve, reject) => {
      geocoder.geocode({ address }, (results, status) => {
        if (status === google.maps.GeocoderStatus.OK && results && results[0]) {
          const location = results[0].geometry.location;
          resolve({
            lat: location.lat(),
            lng: location.lng(),
          });
        } else {
          reject(new Error(`Geocoding failed: ${status}`));
        }
      });
    });
  }

  // Reverse geocode coordinates to address
  async reverseGeocode(location: LocationCoordinates): Promise<string> {
    const google = await this.initialize();
    
    const geocoder = new google.maps.Geocoder();

    return new Promise((resolve, reject) => {
      geocoder.geocode(
        { location: new google.maps.LatLng(location.lat, location.lng) },
        (results, status) => {
          if (status === google.maps.GeocoderStatus.OK && results && results[0]) {
            resolve(results[0].formatted_address);
          } else {
            reject(new Error(`Reverse geocoding failed: ${status}`));
          }
        }
      );
    });
  }

  // Calculate distance between two points using Google Maps API
  async calculateDistance(origin: LocationCoordinates, destination: LocationCoordinates): Promise<{
    distance: string;
    duration: string;
    distanceValue: number;
    durationValue: number;
  }> {
    const google = await this.initialize();
    
    const service = new google.maps.DistanceMatrixService();

    return new Promise((resolve, reject) => {
      service.getDistanceMatrix(
        {
          origins: [new google.maps.LatLng(origin.lat, origin.lng)],
          destinations: [new google.maps.LatLng(destination.lat, destination.lng)],
          travelMode: google.maps.TravelMode.WALKING,
          unitSystem: google.maps.UnitSystem.METRIC,
        },
        (response, status) => {
          if (status === google.maps.DistanceMatrixStatus.OK && response) {
            const element = response.rows[0].elements[0];
            if (element.status === google.maps.DistanceMatrixElementStatus.OK) {
              resolve({
                distance: element.distance.text,
                duration: element.duration.text,
                distanceValue: element.distance.value,
                durationValue: element.duration.value,
              });
            } else {
              reject(new Error(`Distance calculation failed: ${element.status}`));
            }
          } else {
            reject(new Error(`Distance matrix request failed: ${status}`));
          }
        }
      );
    });
  }

  // Get current location using Google Maps Geolocation API (as backup)
  async getCurrentLocationWithMaps(): Promise<LocationCoordinates> {
    await this.initialize();

    return new Promise((resolve, reject) => {
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            resolve({
              lat: position.coords.latitude,
              lng: position.coords.longitude,
            });
          },
          (error) => {
            reject(new Error(`Geolocation error: ${error.message}`));
          }
        );
      } else {
        reject(new Error('Geolocation not supported'));
      }
    });
  }

  // Helper method to convert travel mode
  private getTravelMode(mode: string): google.maps.TravelMode {
    switch (mode.toUpperCase()) {
      case 'DRIVING':
        return google.maps.TravelMode.DRIVING;
      case 'WALKING':
        return google.maps.TravelMode.WALKING;
      case 'BICYCLING':
        return google.maps.TravelMode.BICYCLING;
      case 'TRANSIT':
        return google.maps.TravelMode.TRANSIT;
      default:
        return google.maps.TravelMode.WALKING;
    }
  }

  // Push data to server
  private async pushToServer(payload: ServerDataPayload): Promise<void> {
    console.log('🗺️ Maps Data for Server:', {
      ...payload,
      formattedData: {
        sessionId: payload.sessionId,
        type: payload.type,
        timestamp: new Date(payload.timestamp).toISOString(),
        data: payload.data,
      }
    });

    try {
      if (payload.type === 'route_calculated') {
        const response = await fetch('http://localhost:8000/navigation/route', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            sessionId: payload.sessionId,
            destination_name: (payload.data as RouteResult).destination.address,
            ...payload.data as RouteResult,
          }),
        });

        if (!response.ok) {
          console.warn('Failed to push route data to server:', response.statusText);
        } else {
          const result = await response.json();
          console.log('✅ Route data pushed to server:', result.message);
        }
      } else if (payload.type === 'places_search') {
        // Could add places search endpoint later if needed
        console.log('🔍 Places search result logged (no server endpoint yet)');
      }
    } catch (error) {
      console.error('❌ Error pushing maps data to server:', error);
    }
  }

  // Check if the service is ready
  isReady(): boolean {
    return this.isLoaded;
  }

  // Get the Google Maps API instance (for advanced usage)
  async getGoogleMapsAPI(): Promise<typeof google> {
    return this.initialize();
  }
}

// Export singleton instance
export const mapsService = new MapsService();
export default mapsService; 