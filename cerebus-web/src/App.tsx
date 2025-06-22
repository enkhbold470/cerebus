import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { WebRTCIntegration } from './WebRTCIntegration';
import { GPSMapsDemo } from './components/GPSMapsDemo';
import { WAKE_WORD_SERVER_URL } from '../constants';

import './App.css';

// System state management
type CerebusState = 'idle' | 'initializing' | 'active' | 'error';

interface SystemStatus {
	gps: boolean;
	audio: boolean;
	wakeWord: boolean;
	server: boolean;
}

interface SSEEvent {
	type: string;
	data: Record<string, unknown>;
	timestamp: number;
	client_id: string;
}

function App() {
	const [cerebusState, setCerebusState] = useState<CerebusState>('idle');
	const [systemStatus, setSystemStatus] = useState<SystemStatus>({
		gps: false,
		audio: false,
		wakeWord: false,
		server: false,
	});
	const [statusMessage, setStatusMessage] = useState('Ready to start');
	const [isRecording, setIsRecording] = useState(false);
	const [wakeWordDetected, setWakeWordDetected] = useState(false);
	const [sseConnectionStatus, setSseConnectionStatus] = useState<
		'connecting' | 'connected' | 'error'
	>('connecting');

	// References for cleanup
	const streamRef = useRef<MediaStream | null>(null);
	const mediaRecorderRef = useRef<MediaRecorder | null>(null);

	// Generate client ID
	const clientId = useMemo(
		() => `cerebus-client-${Math.random().toString(36).substr(2, 9)}`,
		[]
	);

	// Wake word detection handler
	const handleWakeWordDetected = useCallback(
		(confidence: number) => {
			console.log(
				`🔥 Wake word detected with ${(confidence * 100).toFixed(
					1
				)}% confidence`
			);
			setWakeWordDetected(true);
			setSystemStatus(prev => ({ ...prev, wakeWord: true })); // Update wake word status to green
			setStatusMessage(
				`Wake word detected (${(confidence * 100).toFixed(1)}% confidence)`
			);

			// Reset after 3 seconds
			setTimeout(() => {
				setWakeWordDetected(false);
				setSystemStatus(prev => ({ ...prev, wakeWord: false })); // Reset wake word status
				setStatusMessage(
					cerebusState === 'active'
						? 'Listening for commands...'
						: 'Ready to start'
				);
			}, 3000);
		},
		[cerebusState]
	);

	// SSE connection setup - always active to detect wake words
	useEffect(() => {
		const eventSource = new EventSource(
			`${WAKE_WORD_SERVER_URL}/sse/${clientId}`
		);

		eventSource.onopen = () => {
			console.log('📡 Connected to Cerebus server');
			setSseConnectionStatus('connected');
			setSystemStatus(prev => ({ ...prev, server: true }));
		};

		eventSource.onmessage = event => {
			try {
				const data = JSON.parse(event.data) as SSEEvent;
				console.log('📡 SSE Event received:', data.type, data);

				// Handle wake word detection
				if (data.type === 'wake_word_detected') {
					const confidence = data.data.confidence as number;
					console.log('🔥 Wake word detected via SSE:', confidence);
					handleWakeWordDetected(confidence);
				}

				// Handle agent responses
				if (data.type === 'agent_start') {
					console.log('🤖 Agent started via SSE');
					setStatusMessage('Agent activated');
					setSystemStatus(prev => ({ ...prev, wakeWord: true }));
				}

				// Handle connection events
				if (data.type === 'connected') {
					console.log('✅ SSE connection confirmed');
				}
			} catch (error) {
				console.error('Error parsing server message:', error);
			}
		};

		eventSource.onerror = error => {
			console.error('📡 Server connection error:', error);
			setSseConnectionStatus('error');
			setSystemStatus(prev => ({ ...prev, server: false }));
		};

		return () => {
			eventSource.close();
		};
	}, [clientId, handleWakeWordDetected]);

	// Initialize audio recording
	const initializeAudio = async (): Promise<boolean> => {
		try {
			const stream = await navigator.mediaDevices.getUserMedia({
				audio: {
					echoCancellation: true,
					noiseSuppression: true,
					autoGainControl: true,
				},
			});

			streamRef.current = stream;

			// Set up MediaRecorder for wake word detection
			const mediaRecorder = new MediaRecorder(stream, {
				mimeType: MediaRecorder.isTypeSupported('audio/webm')
					? 'audio/webm'
					: 'audio/mp4',
			});

			mediaRecorderRef.current = mediaRecorder;
			mediaRecorder.start(100); // 100ms chunks for real-time processing
			setIsRecording(true);

			return true;
		} catch (error) {
			console.error('Error accessing microphone:', error);
			return false;
		}
	};

	// Initialize GPS
	const initializeGPS = async (): Promise<boolean> => {
		return new Promise(resolve => {
			if (!navigator.geolocation) {
				console.error('Geolocation not supported');
				resolve(false);
				return;
			}

			navigator.geolocation.getCurrentPosition(
				position => {
					console.log('📍 GPS initialized:', position.coords);
					resolve(true);
				},
				error => {
					console.error('GPS error:', error);
					resolve(false);
				},
				{ enableHighAccuracy: true, timeout: 10000 }
			);
		});
	};

	// Main function to start all Cerebus systems
	const startCerebus = async () => {
		setCerebusState('initializing');
		setStatusMessage('Initializing systems...');

		try {
			// Initialize GPS
			setStatusMessage('Initializing GPS...');
			const gpsSuccess = await initializeGPS();
			setSystemStatus(prev => ({ ...prev, gps: gpsSuccess }));

			if (!gpsSuccess) {
				throw new Error('GPS initialization failed');
			}

			// Initialize audio
			setStatusMessage('Initializing audio systems...');
			const audioSuccess = await initializeAudio();
			setSystemStatus(prev => ({ ...prev, audio: audioSuccess }));

			if (!audioSuccess) {
				throw new Error('Audio initialization failed');
			}

			// All systems ready
			setCerebusState('active');
			setStatusMessage('All systems active - Listening for "Hey Cerebus"');
		} catch (error) {
			console.error('Cerebus initialization failed:', error);
			setCerebusState('error');
			setStatusMessage(
				`Initialization failed: ${
					error instanceof Error ? error.message : 'Unknown error'
				}`
			);
		}
	};

	// Stop all systems
	const stopCerebus = () => {
		// Stop audio recording
		if (
			mediaRecorderRef.current &&
			mediaRecorderRef.current.state !== 'inactive'
		) {
			mediaRecorderRef.current.stop();
		}

		// Stop audio stream
		if (streamRef.current) {
			streamRef.current.getTracks().forEach(track => track.stop());
			streamRef.current = null;
		}

		setIsRecording(false);
		setCerebusState('idle');
		setSystemStatus({
			gps: false,
			audio: false,
			wakeWord: false,
			server: false,
		});
		setStatusMessage('Ready to start');
		setWakeWordDetected(false);
	};

	// Cleanup on unmount
	useEffect(() => {
		return () => {
			if (streamRef.current) {
				streamRef.current.getTracks().forEach(track => track.stop());
			}
		};
	}, []);

	return (
		<div className="cerebus-app">
			<div className="cerebus-container">
				<header className="cerebus-header">
					<h1 className="cerebus-title">Cerebus</h1>
					<p className="cerebus-subtitle">AI-Powered Navigation Assistant</p>
				</header>

				<main className="cerebus-main">
					{/* System Status */}
					<div className="system-status">
						<div className="status-grid">
							<div
								className={`status-item ${
									systemStatus.server ? 'active' : 'inactive'
								}`}
							>
								<span className="status-icon">🌐</span>
								<span className="status-label">Server</span>
							</div>
							<div
								className={`status-item ${
									systemStatus.gps ? 'active' : 'inactive'
								}`}
							>
								<span className="status-icon">📍</span>
								<span className="status-label">GPS</span>
							</div>
							<div
								className={`status-item ${
									systemStatus.audio ? 'active' : 'inactive'
								}`}
							>
								<span className="status-icon">🎤</span>
								<span className="status-label">Audio</span>
							</div>
							<div
								className={`status-item ${
									systemStatus.wakeWord ? 'active' : 'inactive'
								}`}
							>
								<span className="status-icon">🔥</span>
								<span className="status-label">Wake Word</span>
							</div>
						</div>
					</div>

					{/* Status Message */}
					<div className="status-message">
						<p
							className={`status-text ${
								wakeWordDetected ? 'wake-word-active' : ''
							}`}
						>
							{statusMessage}
						</p>
						<p className="connection-status">
							SSE:{' '}
							{sseConnectionStatus === 'connected'
								? '✅ Connected'
								: sseConnectionStatus === 'error'
								? '❌ Error'
								: '🔄 Connecting...'}
						</p>
					</div>

					{/* Main Action Button */}
					<div className="action-section">
						{cerebusState === 'idle' && (
							<button
								onClick={startCerebus}
								className="cerebus-button primary"
								type="button"
							>
								Start Cerebus
							</button>
						)}

						{cerebusState === 'initializing' && (
							<button className="cerebus-button loading" disabled type="button">
								<span className="loading-spinner"></span>
								Initializing...
							</button>
						)}

						{cerebusState === 'active' && (
							<div className="active-controls">
								<div className="active-indicator">
									<span className="pulse-dot"></span>
									System Active
								</div>
								<button
									onClick={stopCerebus}
									className="cerebus-button secondary"
									type="button"
								>
									Stop System
								</button>
							</div>
						)}

						{cerebusState === 'error' && (
							<div className="error-controls">
								<button
									onClick={() => setCerebusState('idle')}
									className="cerebus-button primary"
									type="button"
								>
									Retry
								</button>
							</div>
						)}
					</div>

					{/* Wake Word Detection Alert */}
					{wakeWordDetected && (
						<div className="wake-word-alert">
							<span className="wake-word-icon">🔥</span>
							<span className="wake-word-text">Wake Word Detected</span>
						</div>
					)}
				</main>

				{/* Hidden WebRTC Integration - still functional but not visible */}
				<div style={{ display: 'none' }}>
					<WebRTCIntegration
						isRecording={isRecording}
						audioStream={streamRef.current}
					/>
				</div>

				{/* Hidden GPS Component - functional but not cluttering UI */}
				<div style={{ display: 'none' }}>
					<GPSMapsDemo />
				</div>
			</div>
		</div>
	);
}

export default App;
