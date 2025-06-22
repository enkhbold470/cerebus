import { useEffect, useState } from 'react';
import { WAKE_WORD_SERVER_URL } from '../constants';

interface SSEEvent {
	type: string;
	data: Record<string, unknown>;
	timestamp: number;
	client_id: string;
}

interface TimestampData {
	timestamp: number;
	datetime: string;
	message: string;
}

const SSETestComponent = () => {
	const [events, setEvents] = useState<SSEEvent[]>([]);
	const [connectionStatus, setConnectionStatus] = useState<
		'connecting' | 'connected' | 'disconnected'
	>('connecting');
	const [clientId] = useState(
		`react-client-${Math.random().toString(36).substr(2, 9)}`,
	);

	useEffect(() => {
		// Connect to SSE endpoint
		const eventSource = new EventSource(
			`${WAKE_WORD_SERVER_URL}/sse/${clientId}`,
		);

		eventSource.onopen = () => {
			console.log('SSE connection opened');
			setConnectionStatus('connected');
		};

		eventSource.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data) as SSEEvent;
				console.log('SSE event received:', data);

				setEvents((prev) => [...prev.slice(-49), data]); // Keep last 50 events
			} catch (error) {
				console.error('Error parsing SSE data:', error);
			}
		};

		eventSource.onerror = (error) => {
			console.error('SSE error:', error);
			setConnectionStatus('disconnected');
		};

		return () => {
			eventSource.close();
		};
	}, [clientId]);

	const getStatusColor = () => {
		switch (connectionStatus) {
			case 'connected':
				return 'text-green-600';
			case 'disconnected':
				return 'text-red-600';
			case 'connecting':
				return 'text-yellow-600';
		}
	};

	const formatEventDisplay = (event: SSEEvent) => {
		switch (event.type) {
			case 'timestamp_test': {
				const timestampData = event.data as unknown as TimestampData;
				return (
					<div className="text-blue-600">
						<strong>Timestamp Test:</strong> {timestampData.datetime}
						<br />
						<small>Unix: {timestampData.timestamp}</small>
					</div>
				);
			}
			case 'wake_word_detected': {
				const confidence = (event.data.confidence as number) ?? 0;
				return (
					<div className="text-red-600 font-bold">
						<strong>WAKE WORD DETECTED!</strong>
						<br />
						Confidence: {confidence.toFixed(3)}
					</div>
				);
			}
			case 'connected': {
				return (
					<div className="text-green-600">
						<strong>Connected</strong> - Client ID:{' '}
						{event.data.client_id as string}
					</div>
				);
			}
			case 'keepalive': {
				return (
					<div className="text-gray-500">
						<small>
							Keepalive -{' '}
							{new Date(event.timestamp * 1000).toLocaleTimeString()}
						</small>
					</div>
				);
			}
			default: {
				return (
					<div>
						<strong>{event.type}:</strong> {JSON.stringify(event.data)}
					</div>
				);
			}
		}
	};

	return (
		<div className="max-w-4xl mx-auto p-6">
			<h1 className="text-2xl font-bold mb-4">SSE Test Component</h1>

			<div className="bg-gray-100 p-4 rounded-lg mb-4">
				<p>
					<strong>Client ID:</strong> {clientId}
				</p>
				<p>
					<strong>Connection Status:</strong>
					<span className={`ml-2 font-semibold ${getStatusColor()}`}>
						{connectionStatus.toUpperCase()}
					</span>
				</p>
				<p>
					<strong>Events Received:</strong> {events.length}
				</p>
			</div>

			<div className="bg-white border rounded-lg h-96 overflow-y-auto p-4">
				<h2 className="text-lg font-semibold mb-3">Live Events</h2>
				{events.length === 0 ? (
					<p className="text-gray-500">Waiting for events...</p>
				) : (
					<div className="space-y-2">
						{events.map((event, index) => (
							<div key={index} className="border-b pb-2 last:border-b-0">
								<div className="text-xs text-gray-500 mb-1">
									{new Date(event.timestamp * 1000).toLocaleTimeString()}
								</div>
								{formatEventDisplay(event)}
							</div>
						))}
					</div>
				)}
			</div>

			<div className="mt-4 text-sm text-gray-600">
				<p>
					<strong>Expected Events:</strong>
				</p>
				<ul className="list-disc list-inside">
					<li>
						<span className="text-blue-600">timestamp_test</span> - Every 1
						second with current timestamp
					</li>
					<li>
						<span className="text-red-600">wake_word_detected</span> - When wake
						word is detected
					</li>
					<li>
						<span className="text-gray-500">keepalive</span> - Every 30 seconds
					</li>
				</ul>
			</div>
		</div>
	);
};

export default SSETestComponent;
