const { Client, TravelMode } = require("@googlemaps/google-maps-services-js");
const mapsClient = new Client({});

/**
 * Calculates the distance between two GPS coordinates in meters.
 * @param {{lat: number, lng: number}} point1 The first point.
 * @param {{lat: number, lng: number}} point2 The second point.
 * @returns {number} The distance in meters.
 */
function getDistance(point1, point2) {
  const R = 6371e3; // Earth's radius in meters
  const lat1 = (point1.lat * Math.PI) / 180;
  const lat2 = (point2.lat * Math.PI) / 180;
  const deltaLat = ((point2.lat - point1.lat) * Math.PI) / 180;
  const deltaLng = ((point2.lng - point1.lng) * Math.PI) / 180;

  const a =
    Math.sin(deltaLat / 2) * Math.sin(deltaLat / 2) +
    Math.cos(lat1) *
      Math.cos(lat2) *
      Math.sin(deltaLng / 2) *
      Math.sin(deltaLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c;
}

/**
 * Fetches walking directions from the Google Maps Directions API.
 * @param {string} origin The starting point (e.g., "lat,lng" or address).
 * @param {string} destination The ending point (e.g., "lat,lng" or address).
 * @param {string} [destinationName] An optional name for the destination.
 * @returns {Promise<object>} A promise that resolves with the formatted route data.
 */
async function getRouteData(origin, destination, destinationName) {
  const request = {
    params: {
      origin: origin,
      destination: destination,
      mode: TravelMode.walking,
      key: process.env.GOOGLE_MAPS_API_KEY,
    },
  };

  const response = await mapsClient.directions(request);

  if (response.data.status !== "OK") {
    throw new Error(`Directions request failed: ${response.data.status}`);
  }

  const route = response.data.routes[0];
  const leg = route.legs[0];

  const startInstruction = {
    instruction: `Starting navigation to ${
      destinationName || "your destination"
    }. Begin walking.`,
    distance: "",
    isStart: true,
  };

  const endInstruction = {
    instruction: `You have arrived at ${destinationName || "your destination"}`,
    distance: "",
    isEnd: true,
  };

  const steps = leg.steps.map((step) => ({
    instruction: step.html_instructions,
    distance: step.distance.text,
    duration: step.duration.text,
    originalInstruction: step.html_instructions,
    isStep: true,
  }));

  return {
    timestamp: new Date().toISOString(),
    origin: {
      lat: leg.start_location.lat,
      lng: leg.start_location.lng,
      address: leg.start_address,
    },
    destination: {
      lat: leg.end_location.lat,
      lng: leg.end_location.lng,
      address: leg.end_address,
    },
    route_summary: {
      distance: leg.distance.text,
      duration: leg.duration.text,
      travel_mode: "WALKING",
    },
    voice_instructions: [startInstruction, ...steps, endInstruction],
    detailed_steps: leg.steps.map((step) => ({
      instruction: step.html_instructions,
      distance: step.distance.text,
      duration: step.duration.text,
      start_location: step.start_location,
      end_location: step.end_location,
    })),
    polyline: route.overview_polyline.points,
  };
}

/**
 * Searches for nearby places using the Google Maps Text Search API.
 * @param {string} location The location to search from (e.g., "lat,lng").
 * @param {string} query The search query (e.g., "coffee shop").
 * @param {number} [radius=5000] The search radius in meters.
 * @returns {Promise<object[]>} A promise that resolves with an array of place results.
 */
async function getNearbyPlaces(location, query, radius = 5000) {
  const request = {
    params: {
      query: query,
      location: location,
      radius: radius,
      key: process.env.GOOGLE_MAPS_API_KEY,
    },
  };

  const response = await mapsClient.textSearch(request);

  if (response.data.status !== "OK") {
    throw new Error(`Places search failed: ${response.data.status}`);
  }

  // Sanitize the results to send back only what's needed
  return response.data.results.map((place) => ({
    place_id: place.place_id,
    name: place.name,
    address: place.formatted_address,
    location: place.geometry.location,
    rating: place.rating,
    user_ratings_total: place.user_ratings_total,
  }));
}

module.exports = {
  getDistance,
  getRouteData,
  getNearbyPlaces,
};
