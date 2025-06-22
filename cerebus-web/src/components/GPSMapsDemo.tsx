import React, { useState } from 'react';
import { useGeolocation } from '../hooks/useGeolocation';
import { useMaps } from '../hooks/useMaps';
import type { LocationData, PlaceResult, RouteResult } from '../types/gps';

export const GPSMapsDemo: React.FC = () => {
	const [destination, setDestination] = useState<string>('');
	const [searchQuery, setSearchQuery] = useState<string>('');
	const [currentRoute, setCurrentRoute] = useState<RouteResult | null>(null);
	const [nearbyPlaces, setNearbyPlaces] = useState<PlaceResult[]>([]);
	const [log, setLog] = useState<string[]>([]);

	// GPS Hook
	const gps = useGeolocation({
		onLocationUpdate: (location: LocationData) => {
			addToLog(
				`📍 Location Update: ${location.lat.toFixed(6)}, ${location.lng.toFixed(
					6
				)} (±${location.accuracy}m)`
			);
		},
		onError: (error: string) => {
			addToLog(`❌ GPS Error: ${error}`);
		},
	});

	// Maps Hook
	const maps = useMaps({
		autoInitialize: true,
		onError: (error: string) => {
			addToLog(`🗺️ Maps Error: ${error}`);
		},
	});

	const addToLog = (message: string) => {
		const timestamp = new Date().toLocaleTimeString();
		setLog(prev => [...prev.slice(-9), `[${timestamp}] ${message}`]);
	};

	const handleGetCurrentLocation = async () => {
		addToLog('🔍 Getting current location...');
		const location = await gps.getCurrentLocation();
		if (location) {
			addToLog(
				`✅ Current location: ${location.lat.toFixed(
					6
				)}, ${location.lng.toFixed(6)}`
			);
		}
	};

	const handleStartTracking = () => {
		addToLog('🚀 Starting GPS tracking...');
		gps.startTracking();
	};

	const handleStopTracking = () => {
		addToLog('🛑 Stopping GPS tracking...');
		gps.stopTracking();
	};

	const handleCalculateRoute = async () => {
		if (!destination.trim()) {
			addToLog('❌ Please enter a destination');
			return;
		}

		if (!gps.location || gps.location.lat === 0) {
			addToLog('❌ Current location not available. Get location first.');
			return;
		}

		addToLog(`🗺️ Calculating route to: ${destination}`);

		try {
			// First geocode the destination
			const destinationCoords = await maps.geocodeAddress(destination);
			if (!destinationCoords) {
				addToLog('❌ Could not find destination coordinates');
				return;
			}

			// Calculate route
			const route = await maps.calculateRoute({
				origin: { lat: gps.location.lat, lng: gps.location.lng },
				destination: destinationCoords,
				destinationName: destination,
				travelMode: 'WALKING',
			});

			if (route) {
				setCurrentRoute(route);
				addToLog(
					`✅ Route calculated: ${route.route_summary.distance}, ${route.route_summary.duration}`
				);
				addToLog(
					`📢 Voice instructions: ${route.voice_instructions.length} steps`
				);
			}
		} catch (error) {
			addToLog(`❌ Route calculation failed: ${error}`);
		}
	};

	const handleSearchNearby = async () => {
		if (!searchQuery.trim()) {
			addToLog('❌ Please enter a search query');
			return;
		}

		if (!gps.location || gps.location.lat === 0) {
			addToLog('❌ Current location not available. Get location first.');
			return;
		}

		addToLog(`🔍 Searching for: ${searchQuery}`);

		const places = await maps.searchNearbyPlaces({
			location: { lat: gps.location.lat, lng: gps.location.lng },
			query: searchQuery,
			radius: 2000,
		});

		setNearbyPlaces(places);
		addToLog(`✅ Found ${places.length} places`);
	};

	const handleGetAddress = async () => {
		if (!gps.location || gps.location.lat === 0) {
			addToLog('❌ Current location not available. Get location first.');
			return;
		}

		addToLog('🏠 Getting current address...');
		const address = await maps.reverseGeocode({
			lat: gps.location.lat,
			lng: gps.location.lng,
		});

		if (address) {
			addToLog(`📍 Address: ${address}`);
		}
	};

	const handleRequestPermission = async () => {
		addToLog('🔐 Requesting location permission...');
		const granted = await gps.requestPermission();
		addToLog(granted ? '✅ Permission granted' : '❌ Permission denied');
	};

	return (
		<div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
			<h2>🧭 GPS & Maps Integration Demo</h2>

			{/* Status Section */}
			<div
				style={{
					backgroundColor: '#f5f5f5',
					padding: '15px',
					borderRadius: '8px',
					marginBottom: '20px',
				}}
			>
				<h3>📊 Status</h3>
				<div
					style={{
						display: 'grid',
						gridTemplateColumns: '1fr 1fr',
						gap: '10px',
					}}
				>
					<div>
						<strong>GPS:</strong> {gps.permission} |{' '}
						{gps.isTracking ? '🟢 Tracking' : '🔴 Not tracking'}
						{gps.isLoading && ' | 🔄 Loading...'}
					</div>
					<div>
						<strong>Maps:</strong>{' '}
						{maps.isInitialized ? '🟢 Ready' : '🔴 Not ready'}
						{maps.isLoading && ' | 🔄 Loading...'}
					</div>
				</div>
				{gps.location && gps.location.lat !== 0 && (
					<div style={{ marginTop: '10px' }}>
						<strong>Current Location:</strong> {gps.location.lat.toFixed(6)},{' '}
						{gps.location.lng.toFixed(6)}
						{gps.location.accuracy &&
							` (±${gps.location.accuracy.toFixed(0)}m)`}
					</div>
				)}
			</div>

			{/* GPS Controls */}
			<div style={{ marginBottom: '20px' }}>
				<h3>📍 GPS Controls</h3>
				<div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
					<button onClick={handleRequestPermission} disabled={gps.isLoading}>
						🔐 Request Permission
					</button>
					<button onClick={handleGetCurrentLocation} disabled={gps.isLoading}>
						📍 Get Current Location
					</button>
					<button
						onClick={handleStartTracking}
						disabled={gps.isTracking || gps.isLoading}
					>
						🚀 Start Tracking
					</button>
					<button onClick={handleStopTracking} disabled={!gps.isTracking}>
						🛑 Stop Tracking
					</button>
				</div>
			</div>

			{/* Maps Controls */}
			<div style={{ marginBottom: '20px' }}>
				<h3>🗺️ Maps Controls</h3>

				<div style={{ marginBottom: '15px' }}>
					<label style={{ display: 'block', marginBottom: '5px' }}>
						<strong>Calculate Route:</strong>
					</label>
					<div style={{ display: 'flex', gap: '10px' }}>
						<input
							type="text"
							value={destination}
							onChange={e => setDestination(e.target.value)}
							placeholder="Enter destination (e.g., Times Square, New York)"
							style={{
								flex: 1,
								padding: '8px',
								borderRadius: '4px',
								border: '1px solid #ccc',
							}}
						/>
						<button onClick={handleCalculateRoute} disabled={maps.isLoading}>
							🗺️ Calculate Route
						</button>
					</div>
				</div>

				<div style={{ marginBottom: '15px' }}>
					<label style={{ display: 'block', marginBottom: '5px' }}>
						<strong>Search Nearby:</strong>
					</label>
					<div style={{ display: 'flex', gap: '10px' }}>
						<input
							type="text"
							value={searchQuery}
							onChange={e => setSearchQuery(e.target.value)}
							placeholder="Search for places (e.g., restaurants, gas stations)"
							style={{
								flex: 1,
								padding: '8px',
								borderRadius: '4px',
								border: '1px solid #ccc',
							}}
						/>
						<button onClick={handleSearchNearby} disabled={maps.isLoading}>
							🔍 Search
						</button>
					</div>
				</div>

				<button onClick={handleGetAddress} disabled={maps.isLoading}>
					🏠 Get Current Address
				</button>
			</div>

			{/* Results */}
			{currentRoute && (
				<div style={{ marginBottom: '20px' }}>
					<h3>🛣️ Current Route</h3>
					<div
						style={{
							backgroundColor: '#f0f8ff',
							padding: '15px',
							borderRadius: '8px',
							border: '1px solid #cce7ff',
						}}
					>
						<p>
							<strong>To:</strong> {currentRoute.destination.address}
						</p>
						<p>
							<strong>Distance:</strong> {currentRoute.route_summary.distance}
						</p>
						<p>
							<strong>Duration:</strong> {currentRoute.route_summary.duration}
						</p>
						<p>
							<strong>Instructions:</strong>{' '}
							{currentRoute.voice_instructions.length} steps
						</p>
						<details>
							<summary style={{ cursor: 'pointer', marginTop: '10px' }}>
								📢 View Voice Instructions
							</summary>
							<ol style={{ marginTop: '10px', paddingLeft: '20px' }}>
								{currentRoute.voice_instructions.map((instruction, index) => (
									<li key={index} style={{ marginBottom: '5px' }}>
										{instruction.instruction}
										{instruction.distance && instruction.duration && (
											<span style={{ color: '#666', fontSize: '0.9em' }}>
												{' '}
												({instruction.distance}, {instruction.duration})
											</span>
										)}
									</li>
								))}
							</ol>
						</details>
					</div>
				</div>
			)}

			{nearbyPlaces.length > 0 && (
				<div style={{ marginBottom: '20px' }}>
					<h3>📍 Nearby Places ({nearbyPlaces.length})</h3>
					<div
						style={{
							maxHeight: '300px',
							overflowY: 'auto',
							border: '1px solid #ddd',
							borderRadius: '8px',
						}}
					>
						{nearbyPlaces.slice(0, 10).map((place, index) => (
							<div
								key={place.place_id}
								style={{
									padding: '10px',
									borderBottom: index < 9 ? '1px solid #eee' : 'none',
								}}
							>
								<div style={{ fontWeight: 'bold' }}>{place.name}</div>
								<div style={{ fontSize: '0.9em', color: '#666' }}>
									{place.address}
								</div>
								{place.rating && (
									<div
										style={{
											fontSize: '0.8em',
											color: '#888',
											marginTop: '5px',
										}}
									>
										⭐ {place.rating} ({place.user_ratings_total} reviews)
									</div>
								)}
							</div>
						))}
					</div>
				</div>
			)}

			{/* Activity Log */}
			<div>
				<h3>📝 Activity Log</h3>
				<div
					style={{
						height: '200px',
						overflowY: 'auto',
						backgroundColor: '#1a1a1a',
						color: '#00ff00',
						padding: '10px',
						borderRadius: '8px',
						fontFamily: 'monospace',
						fontSize: '0.9em',
					}}
				>
					{log.length === 0 ? (
						<div style={{ color: '#666' }}>No activity yet...</div>
					) : (
						log.map((entry, index) => (
							<div key={index} style={{ marginBottom: '2px' }}>
								{entry}
							</div>
						))
					)}
				</div>
			</div>

			{/* Instructions */}
			<div
				style={{
					marginTop: '20px',
					padding: '15px',
					backgroundColor: '#fff3cd',
					borderRadius: '8px',
					border: '1px solid #ffeaa7',
				}}
			>
				<h4>📖 Instructions:</h4>
				<ol>
					<li>
						First, request location permission and get your current location
					</li>
					<li>
						Try calculating a route to a destination (e.g., "Central Park, New
						York")
					</li>
					<li>
						Search for nearby places (e.g., "coffee shops", "restaurants")
					</li>
					<li>
						Check the console (F12) to see the data structures that would be
						sent to the server
					</li>
					<li>
						All GPS updates and route calculations are logged to console for
						server integration
					</li>
				</ol>
			</div>
		</div>
	);
};
