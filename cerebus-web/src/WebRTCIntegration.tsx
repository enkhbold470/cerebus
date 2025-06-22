import { useState, useRef, useEffect } from 'react';

interface ServerEvent {
	type: string;
	data: Record<string, unknown>;
	timestamp: number;
	client_id: string;
}

interface WebRTCIntegrationProps {
	onWakeWordDetected: (confidence: number) => void;
	onAudioLevel: (level: number) => void;
	onServerEvent: (event: ServerEvent) => void;
	isRecording: boolean;
	audioStream: MediaStream | null;
}

const WEBSOCKET_SERVER_URL = 'ws://localhost:8000';
const HTTP_SERVER_URL = 'http://localhost:8000';

export function WebRTCIntegration({
	onWakeWordDetected,
	onAudioLevel,
	onServerEvent,
	isRecording,
	audioStream,
}: WebRTCIntegrationProps) {
	const [webrtcConnected, setWebrtcConnected] = useState(false);
	const [sseConnected, setSseConnected] = useState(false);
	const [serverEvents, setServerEvents] = useState<ServerEvent[]>([]);

	// Generate unique client ID
	const [clientId] = useState(() => {
		const timestamp = Date.now();
		const random = Math.random().toString(36).substring(2, 8);
		return `client_${timestamp}_${random}`;
	});

	// References for WebSocket and SSE connections
	const webSocketRef = useRef<WebSocket | null>(null);
	const eventSourceRef = useRef<EventSource | null>(null);
	const audioContextRef = useRef<AudioContext | null>(null);
	const processorRef = useRef<ScriptProcessorNode | null>(null);

	// Audio buffering for 2-second chunks
	const audioBufferRef = useRef<Int16Array[]>([]);
	const targetSamplesPerChunk = 32000; // 2 seconds at 16kHz
	const currentSampleCountRef = useRef(0);

	// Connect to SSE endpoint
	const connectSSE = () => {
		if (!clientId || sseConnected) return;

		const eventSource = new EventSource(`${HTTP_SERVER_URL}/sse/${clientId}`);
		eventSourceRef.current = eventSource;

		eventSource.onopen = () => {
			console.log('✅ SSE Connected');
			setSseConnected(true);
		};

		eventSource.onmessage = (event) => {
			try {
				const serverEvent: ServerEvent = JSON.parse(event.data);
				console.log('📡 SSE event received:', serverEvent);

				// Add to server events list
				setServerEvents((prev) => [...prev.slice(-49), serverEvent]);

				// Handle specific event types
				if (serverEvent.type === 'wake_word_detected') {
					const confidence = serverEvent.data.confidence as number;
					onWakeWordDetected(confidence);
				} else if (serverEvent.type === 'audio_level') {
					const level = serverEvent.data.level as number;
					onAudioLevel(level);
				}

				// Notify parent component
				onServerEvent(serverEvent);
			} catch (error) {
				console.error('Error parsing SSE data:', error);
			}
		};

		eventSource.onerror = (error) => {
			console.error('❌ SSE Error:', error);
			setSseConnected(false);
		};
	};

	// Connect to WebSocket endpoint
	const connectWebSocket = () => {
		if (!clientId || webrtcConnected) return;

		const websocket = new WebSocket(
			`${WEBSOCKET_SERVER_URL}/ws/audio/${clientId}`,
		);
		webSocketRef.current = websocket;

		websocket.onopen = () => {
			console.log('✅ WebSocket Connected');
			setWebrtcConnected(true);
		};

		websocket.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				console.log('📡 WebSocket response:', data);
			} catch (error) {
				console.error('Error parsing WebSocket data:', error);
			}
		};

		websocket.onerror = (error) => {
			console.error('❌ WebSocket Error:', error);
			setWebrtcConnected(false);
		};

		websocket.onclose = () => {
			console.log('🔌 WebSocket Disconnected');
			setWebrtcConnected(false);
		};
	};

	// Send accumulated audio buffer
	const sendAudioChunk = () => {
		if (
			!webSocketRef.current ||
			webSocketRef.current.readyState !== WebSocket.OPEN
		) {
			return;
		}

		if (audioBufferRef.current.length === 0) {
			return;
		}

		// Concatenate all buffers
		const totalSamples = audioBufferRef.current.reduce(
			(sum, buffer) => sum + buffer.length,
			0,
		);
		const combinedBuffer = new Int16Array(totalSamples);

		let offset = 0;
		for (const buffer of audioBufferRef.current) {
			combinedBuffer.set(buffer, offset);
			offset += buffer.length;
		}

		// Convert to ArrayBuffer for sending
		const arrayBuffer = combinedBuffer.buffer.slice(
			combinedBuffer.byteOffset,
			combinedBuffer.byteOffset + combinedBuffer.byteLength,
		);

		console.log(
			`🎵 Sending 2-second audio chunk: ${arrayBuffer.byteLength} bytes (${totalSamples} samples)`,
		);
		webSocketRef.current.send(arrayBuffer);

		// Reset buffer
		audioBufferRef.current = [];
		currentSampleCountRef.current = 0;
	};

	// Set up audio processing and streaming
	const setupAudioProcessing = () => {
		if (!audioStream || !webSocketRef.current || !webrtcConnected) return;

		try {
			const audioContext = new AudioContext();
			const source = audioContext.createMediaStreamSource(audioStream);
			// Use larger buffer size for better performance, but still accumulate for 2-second chunks
			const processor = audioContext.createScriptProcessor(16384, 1, 1);

			audioContextRef.current = audioContext;
			processorRef.current = processor;

			processor.onaudioprocess = (event) => {
				const inputData = event.inputBuffer.getChannelData(0);

				// Convert float32 to int16
				const int16Buffer = new Int16Array(inputData.length);
				for (let i = 0; i < inputData.length; i++) {
					int16Buffer[i] = Math.max(
						-32768,
						Math.min(32767, Math.floor(inputData[i] * 32768)),
					);
				}

				// Add to accumulation buffer
				audioBufferRef.current.push(int16Buffer);
				currentSampleCountRef.current += int16Buffer.length;

				// Send when we have enough samples for 2 seconds
				if (currentSampleCountRef.current >= targetSamplesPerChunk) {
					sendAudioChunk();
				}
			};

			source.connect(processor);
			processor.connect(audioContext.destination);

			console.log('🎵 Audio processing started (2-second chunks)');
		} catch (error) {
			console.error('Error setting up audio processing:', error);
		}
	};

	// Cleanup audio processing
	const cleanupAudioProcessing = () => {
		// Send any remaining audio data
		if (audioBufferRef.current.length > 0) {
			sendAudioChunk();
		}

		if (processorRef.current) {
			processorRef.current.disconnect();
			processorRef.current = null;
		}

		if (audioContextRef.current) {
			audioContextRef.current.close();
			audioContextRef.current = null;
		}

		// Reset audio buffer
		audioBufferRef.current = [];
		currentSampleCountRef.current = 0;

		console.log('🔇 Audio processing stopped');
	};

	// Connect when client ID is available
	useEffect(() => {
		if (clientId) {
			connectSSE();
			connectWebSocket();
		}

		return () => {
			// Cleanup on unmount
			if (eventSourceRef.current) {
				eventSourceRef.current.close();
			}
			if (webSocketRef.current) {
				webSocketRef.current.close();
			}
			cleanupAudioProcessing();
		};
	}, [clientId]);

	// Handle recording state changes
	useEffect(() => {
		if (isRecording && audioStream && webrtcConnected) {
			setupAudioProcessing();
		} else {
			cleanupAudioProcessing();
		}

		return () => {
			cleanupAudioProcessing();
		};
	}, [isRecording, audioStream, webrtcConnected]);

	return (
		<div className="webrtc-integration">
			<h3>🌐 WebRTC & SSE Status</h3>
			<div className="connection-status">
				<div
					className={`status-item ${
						webrtcConnected ? 'connected' : 'disconnected'
					}`}
				>
					WebSocket: {webrtcConnected ? '✅ Connected' : '❌ Disconnected'}
				</div>
				<div
					className={`status-item ${
						sseConnected ? 'connected' : 'disconnected'
					}`}
				>
					SSE: {sseConnected ? '✅ Connected' : '❌ Disconnected'}
				</div>
				<div className="client-id">
					Client ID: <code>{clientId}</code>
				</div>
			</div>

			{serverEvents.length > 0 && (
				<div className="server-events">
					<h4>📡 Server Events ({serverEvents.length})</h4>
					<div
						className="events-list"
						style={{ maxHeight: '200px', overflowY: 'auto' }}
					>
						{serverEvents.slice(-10).map((event, index) => (
							<div key={index} className="event-item">
								<strong>{event.type}</strong>
								<span className="event-time">
									{new Date(event.timestamp * 1000).toLocaleTimeString()}
								</span>
								{event.type === 'audio_level' && (
									<span className="event-data">
										Level: {(event.data.level as number)?.toFixed(0)}
									</span>
								)}
								{event.type === 'wake_word_detected' && (
									<span className="event-data">
										Confidence:{' '}
										{((event.data.confidence as number) * 100)?.toFixed(1)}%
									</span>
								)}
							</div>
						))}
					</div>
				</div>
			)}

			<div className="controls">
				<button
					onClick={connectSSE}
					disabled={sseConnected}
					className="connect-btn"
				>
					{sseConnected ? '✅ SSE Connected' : '🔌 Connect SSE'}
				</button>
				<button
					onClick={connectWebSocket}
					disabled={webrtcConnected}
					className="connect-btn"
				>
					{webrtcConnected ? '✅ WebSocket Connected' : '🔌 Connect WebSocket'}
				</button>
			</div>
		</div>
	);
}
