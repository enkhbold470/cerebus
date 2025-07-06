import { useState, useRef, useEffect } from 'react';

interface WebRTCIntegrationProps {
	isRecording: boolean;
	audioStream: MediaStream | null;
}

const WEBSOCKET_SERVER_URL = 'ws://localhost:8000';

export function WebRTCIntegration({
	isRecording,
	audioStream,
}: WebRTCIntegrationProps) {
	const [webrtcConnected, setWebrtcConnected] = useState(false);

	// Generate unique client ID
	const [clientId] = useState(() => {
		const timestamp = Date.now();
		const random = Math.random().toString(36).substring(2, 8);
		return `client_${timestamp}_${random}`;
	});

	// References for WebSocket connection
	const webSocketRef = useRef<WebSocket | null>(null);
	const audioContextRef = useRef<AudioContext | null>(null);
	const processorRef = useRef<ScriptProcessorNode | null>(null);

	// Audio buffering for 2-second chunks
	const audioBufferRef = useRef<Int16Array[]>([]);
	const targetSamplesPerChunk = 32000; // 2 seconds at 16kHz
	const currentSampleCountRef = useRef(0);

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

			console.log('🎵 Audio processing setup complete');
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
			connectWebSocket();
		}

		return () => {
			// Cleanup on unmount
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
			<h3>🌐 WebRTC Status</h3>
			<div className="connection-status">
				<div
					className={`status-item ${
						webrtcConnected ? 'connected' : 'disconnected'
					}`}
				>
					WebSocket: {webrtcConnected ? '✅ Connected' : '❌ Disconnected'}
				</div>
				<div className="client-id">
					Client ID: <code>{clientId}</code>
				</div>
			</div>

			<div className="controls">
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
