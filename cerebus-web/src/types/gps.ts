// GPS and Location Types
export interface LocationCoordinates {
  lat: number;
  lng: number;
}

export interface LocationData extends LocationCoordinates {
  accuracy?: number;
  timestamp: number;
  speed?: number;
  heading?: number;
}

export interface GeolocationResult {
  location: LocationData;
  error?: string;
  isLoading: boolean;
}

// Route Types
export interface RouteRequest {
  origin: LocationCoordinates;
  destination: LocationCoordinates;
  destinationName?: string;
  travelMode?: 'DRIVING' | 'WALKING' | 'BICYCLING' | 'TRANSIT';
}

export interface RouteStep {
  instruction: string;
  distance: string;
  duration: string;
  originalInstruction?: string;
  isStart?: boolean;
  isEnd?: boolean;
  isStep?: boolean;
}

export interface RouteResult {
  timestamp: string;
  origin: {
    lat: number;
    lng: number;
    address: string;
  };
  destination: {
    lat: number;
    lng: number;
    address: string;
  };
  route_summary: {
    distance: string;
    duration: string;
    travel_mode: string;
  };
  voice_instructions: RouteStep[];
  detailed_steps: Array<{
    instruction: string;
    distance: string;
    duration: string;
    start_location: LocationCoordinates;
    end_location: LocationCoordinates;
  }>;
  polyline: string;
}

// Places Search Types
export interface NearbySearchRequest {
  location: LocationCoordinates;
  query: string;
  radius?: number;
}

export interface PlaceResult {
  place_id: string;
  name: string;
  address: string;
  location: LocationCoordinates;
  rating?: number;
  user_ratings_total?: number;
  types?: string[];
}

// Session Data Types
export interface GPSSession {
  sessionId: string;
  startTime: number;
  locationHistory: LocationData[];
  currentLocation?: LocationData;
  isTracking: boolean;
}

export interface NavigationSession {
  sessionId: string;
  startTime: number;
  route?: RouteResult;
  currentStep?: number;
  isActive: boolean;
  destination?: {
    name: string;
    coordinates: LocationCoordinates;
  };
}

// Data for server push
export interface ServerDataPayload {
  type: 'gps_update' | 'route_calculated' | 'places_search' | 'navigation_update';
  timestamp: number;
  sessionId: string;
  data: LocationData | RouteResult | PlaceResult[] | NavigationSession;
} 