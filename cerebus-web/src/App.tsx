import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import reactLogo from './assets/react.svg';
import viteLogo from '/vite.svg';
import { WebRTCIntegration } from './WebRTCIntegration';
import { GPSMapsDemo } from './components/GPSMapsDemo';
import { WAKE_WORD_SERVER_URL } from '../constants';

import './App.css';

interface Recording {
	url: string;
	name: string;
	timestamp: number;
	duration: number;
}

interface AudioChunk {
	blob: Blob;
	timestamp: number;
	chunkIndex: number;
}

interface TranscriptionResult {
	text: string;
	timestamp: number;
	chunkIndex: number;
}

interface ServerEvent {
	type: string;
	data: Record<string, unknown>;
	timestamp: number;
	client_id: string;
}

interface SSEEvent {
	type: string;
	data: Record<string, unknown>;
	timestamp: number;
	client_id: string;
}

// Throttle utility function
const throttle = <T extends unknown[]>(
	func: (...args: T) => void,
	delay: number,
) => {
	let timeoutId: number | null = null;
	let lastExecTime = 0;
	return (...args: T) => {
		const currentTime = Date.now();

		if (currentTime - lastExecTime > delay) {
			func(...args);
			lastExecTime = currentTime;
		} else {
			if (timeoutId) clearTimeout(timeoutId);
			timeoutId = setTimeout(() => {
				func(...args);
				lastExecTime = Date.now();
			}, delay - (currentTime - lastExecTime));
		}
	};
};

function App() {
	const [recordings, setRecordings] = useState<Recording[]>([]);
	const [isRecording, setIsRecording] = useState(false);
	const [recordingTime, setRecordingTime] = useState(0);
	const [audioLevel, setAudioLevel] = useState(0);
	const [isPaused, setIsPaused] = useState(false);
	const [isUploading, setIsUploading] = useState(false);

	// Chunk recording states
	const [chunkInterval, setChunkInterval] = useState(2000); // 2 seconds default
	const [streamingMode, setStreamingMode] = useState(false); // Continuous streaming mode
	const [audioChunks, setAudioChunks] = useState<AudioChunk[]>([]);
	const [transcriptionResults, setTranscriptionResults] = useState<
		TranscriptionResult[]
	>([]);
	const [chunkCounter, setChunkCounter] = useState(0);
	const [wakeWordDetected, setWakeWordDetected] = useState(false);

	// Real-time server data from WebRTC
	const [realTimeAudioLevel, setRealTimeAudioLevel] = useState(0);
	const [serverWakeWordDetected, setServerWakeWordDetected] = useState(false);

	// SSE connection state
	const [sseConnected, setSseConnected] = useState(false);
	const [sseEvents, setSseEvents] = useState<SSEEvent[]>([]);
	const [lastTimestamp, setLastTimestamp] = useState<string>('');
	const clientId = useMemo(
		() => `react-client-${Math.random().toString(36).substr(2, 9)}`,
		[],
	);

	const mediaRecorderRef = useRef<MediaRecorder | null>(null);
	const audioChunksRef = useRef<BlobPart[]>([]);
	const streamRef = useRef<MediaStream | null>(null);
	const timerRef = useRef<number | null>(null);
	const analyserRef = useRef<AnalyserNode | null>(null);
	const animationFrameRef = useRef<number | null>(null);

	// Throttled audio level setters to prevent excessive re-renders
	const throttledSetAudioLevel = useCallback(
		throttle((level: number) => setAudioLevel(level), 100), // Update only 10 times per second
		[],
	);

	const throttledSetRealTimeAudioLevel = useCallback(
		throttle((level: number) => setRealTimeAudioLevel(level), 200), // Update only 5 times per second
		[],
	);

	// Cleanup function
	const cleanupRecording = useCallback(() => {
		if (timerRef.current) {
			clearInterval(timerRef.current);
			timerRef.current = null;
		}

		if (animationFrameRef.current) {
			cancelAnimationFrame(animationFrameRef.current);
			animationFrameRef.current = null;
		}

		if (streamRef.current) {
			streamRef.current.getTracks().forEach((track) => track.stop());
			streamRef.current = null;
		}

		setRecordingTime(0);
		setAudioLevel(0);
		setIsPaused(false);
	}, []);

	// Audio level monitoring for visual feedback
	const monitorAudioLevel = useCallback(() => {
		if (!analyserRef.current) return;

		const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
		analyserRef.current.getByteFrequencyData(dataArray);

		const average =
			dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
		throttledSetAudioLevel(average);

		if (isRecording && !isPaused) {
			animationFrameRef.current = requestAnimationFrame(monitorAudioLevel);
		}
	}, [isRecording, isPaused, throttledSetAudioLevel]);

	const startRecording = async () => {
		try {
			const stream = await navigator.mediaDevices.getUserMedia({
				audio: {
					echoCancellation: true,
					noiseSuppression: true,
					autoGainControl: true,
				},
			});

			streamRef.current = stream;

			// Set up audio analysis for visual feedback
			const audioContext = new AudioContext();
			const source = audioContext.createMediaStreamSource(stream);
			const analyser = audioContext.createAnalyser();
			analyser.fftSize = 256;
			source.connect(analyser);
			analyserRef.current = analyser;

			// Set up MediaRecorder
			const mediaRecorder = new MediaRecorder(stream, {
				mimeType: MediaRecorder.isTypeSupported('audio/webm')
					? 'audio/webm'
					: 'audio/mp4',
			});

			mediaRecorderRef.current = mediaRecorder;
			audioChunksRef.current = [];

			mediaRecorder.ondataavailable = (event) => {
				if (event.data.size > 0) {
					audioChunksRef.current.push(event.data);

					// Always process chunk for wake word detection
					processAudioChunk(event);
				}
			};

			mediaRecorder.onstop = async () => {
				const mimeType = MediaRecorder.isTypeSupported('audio/webm')
					? 'audio/webm'
					: 'audio/mp4';
				const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
				const url = URL.createObjectURL(audioBlob);
				const timestamp = Date.now();
				const recordingDuration = recordingTime;
				const name = `recording-${new Date()
					.toLocaleString()
					.replace(/[/,:]/g, '-')}`;

				const newRecording: Recording = {
					url,
					name,
					timestamp,
					duration: recordingDuration,
				};

				setRecordings((prev) => [...prev, newRecording]);

				setIsUploading(true);

				cleanupRecording();
			};

			// Start recording with streaming or chunk interval
			const effectiveInterval = streamingMode ? 100 : chunkInterval; // 100ms for streaming
			console.log(
				`🎤 Starting MediaRecorder with ${effectiveInterval}ms interval (${
					streamingMode ? 'streaming' : 'chunked'
				} mode)`,
			);
			mediaRecorder.start(effectiveInterval);
			setIsRecording(true);

			// Start timer
			timerRef.current = setInterval(() => {
				setRecordingTime((prev) => prev + 1);
			}, 1000);

			// Start audio monitoring
			monitorAudioLevel();
		} catch (error) {
			console.error('Error starting recording:', error);
			alert(
				'Could not access microphone. Please check permissions and ensure you are using HTTPS.',
			);
		}
	};

	// WebRTC event handlers - memoized to prevent recreating on every render
	const handleWakeWordDetected = useCallback(
		(confidence: number) => {
			console.log(
				`🔥 Wake word detected with ${(confidence * 100).toFixed(
					1,
				)}% confidence`,
			);

			// Add result to transcription results
			const transcriptionResult: TranscriptionResult = {
				text: `🔥 WAKE WORD DETECTED: Server (${(confidence * 100).toFixed(
					1,
				)}%)`,
				timestamp: Date.now(),
				chunkIndex: chunkCounter,
			};

			setTranscriptionResults((prev) => [...prev, transcriptionResult]);
			setWakeWordDetected(true);
			setServerWakeWordDetected(true);

			// Reset after 5 seconds
			setTimeout(() => {
				setWakeWordDetected(false);
				setServerWakeWordDetected(false);
			}, 5000);
		},
		[chunkCounter],
	);

	const handleAudioLevel = useCallback(
		(level: number) => {
			throttledSetRealTimeAudioLevel(level);
		},
		[throttledSetRealTimeAudioLevel],
	);

	const handleServerEvent = useCallback((event: ServerEvent) => {
		console.log('📡 Server event received:', event);

		// Update chunk counter for display purposes
		if (event.type === 'audio_level') {
			setChunkCounter((prev) => prev + 1);
		}
	}, []);

	const processAudioChunk = async (event: BlobEvent) => {
		if (event.data.size > 0) {
			const chunk: AudioChunk = {
				blob: event.data,
				timestamp: Date.now(),
				chunkIndex: chunkCounter,
			};

			if (streamingMode) {
				console.log(
					`🌊 Streaming chunk ${chunkCounter}: ${(
						event.data.size / 1024
					).toFixed(2)} KB`,
				);
			} else {
				console.log(
					`📦 Processing audio chunk ${chunkCounter}: ${(
						event.data.size / 1024
					).toFixed(2)} KB`,
				);
			}

			setAudioChunks((prev) => [...prev, chunk]);
			setChunkCounter((prev) => prev + 1);

			// Audio chunks are now handled by WebRTC integration
			// The WebRTC component will stream audio directly to the server
			console.log('📡 Audio chunk will be processed by WebRTC integration');
		}
	};

	const stopRecording = useCallback(() => {
		if (
			mediaRecorderRef.current &&
			mediaRecorderRef.current.state !== 'inactive'
		) {
			mediaRecorderRef.current.stop();
			setIsRecording(false);
		}
	}, []);

	const pauseRecording = useCallback(() => {
		if (
			mediaRecorderRef.current &&
			mediaRecorderRef.current.state === 'recording'
		) {
			mediaRecorderRef.current.pause();
			setIsPaused(true);
			if (timerRef.current) {
				clearInterval(timerRef.current);
			}
		}
	}, []);

	const resumeRecording = useCallback(() => {
		if (
			mediaRecorderRef.current &&
			mediaRecorderRef.current.state === 'paused'
		) {
			mediaRecorderRef.current.resume();
			setIsPaused(false);
			// Resume timer
			timerRef.current = setInterval(() => {
				setRecordingTime((prev) => prev + 1);
			}, 1000);
			// Resume audio monitoring
			monitorAudioLevel();
		}
	}, [monitorAudioLevel]);

	const deleteRecording = useCallback((index: number) => {
		setRecordings((prev) => {
			const updated = [...prev];
			URL.revokeObjectURL(updated[index].url); // Clean up memory
			updated.splice(index, 1);
			return updated;
		});
	}, []);

	// Format time display - memoized to prevent unnecessary recalculations
	const formatTime = useCallback((seconds: number) => {
		const mins = Math.floor(seconds / 60);
		const secs = seconds % 60;
		return `${mins.toString().padStart(2, '0')}:${secs
			.toString()
			.padStart(2, '0')}`;
	}, []);

	// Memoized computed values
	const recentSseEvents = useMemo(
		() => sseEvents.slice(-5).reverse(),
		[sseEvents],
	);

	const recentAudioChunks = useMemo(() => audioChunks.slice(-5), [audioChunks]);

	const audioLevelPercentage = useMemo(
		() => (audioLevel / 255) * 100,
		[audioLevel],
	);

	const serverAudioLevelPercentage = useMemo(
		() => Math.min((realTimeAudioLevel / 2000) * 100, 100),
		[realTimeAudioLevel],
	);

	// Throttled SSE event handler to prevent excessive re-renders
	const throttledSetSseEvents = useCallback(
		throttle((newEvent: SSEEvent) => {
			setSseEvents((prev) => [...prev.slice(-19), newEvent]); // Keep last 20 events
		}, 50), // Update max 20 times per second
		[],
	);

	// SSE connection setup
	useEffect(() => {
		const eventSource = new EventSource(
			`${WAKE_WORD_SERVER_URL}/sse/${clientId}`,
		);

		eventSource.onopen = () => {
			console.log('📡 SSE connection opened');
			setSseConnected(true);
		};

		eventSource.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data) as SSEEvent;
				console.log('📡 SSE event received:', data);

				throttledSetSseEvents(data);

				// Handle specific event types
				if (data.type === 'wake_word_detected') {
					const confidence = data.data.confidence as number;
					handleWakeWordDetected(confidence);
				} else if (data.type === 'audio_level') {
					const level = data.data.level as number;
					handleAudioLevel(level);
				}

				// Handle server events
				handleServerEvent({
					type: data.type,
					data: data.data,
					timestamp: data.timestamp,
					client_id: data.client_id,
				});

				// Update timestamp for display
				if (data.type === 'timestamp_test') {
					const timestampData = data.data as { datetime?: string };
					if (timestampData.datetime) {
						setLastTimestamp(timestampData.datetime);
					}
				}
			} catch (error) {
				console.error('Error parsing SSE data:', error);
			}
		};

		eventSource.onerror = (error) => {
			console.error('📡 SSE error:', error);
			setSseConnected(false);
		};

		return () => {
			eventSource.close();
		};
	}, [clientId, throttledSetSseEvents]);

	// Cleanup on component unmount
	useEffect(() => {
		return () => {
			cleanupRecording();
		};
	}, [cleanupRecording]);

	// Cleanup animation frame when not recording
	useEffect(() => {
		if (!isRecording || isPaused) {
			if (animationFrameRef.current) {
				cancelAnimationFrame(animationFrameRef.current);
				animationFrameRef.current = null;
			}
		}
	}, [isRecording, isPaused]);

	return (
		<>
			<div>
				<a href="https://vite.dev" target="_blank">
					<img src={viteLogo} className="logo" alt="Vite logo" />
				</a>
				<a href="https://react.dev" target="_blank">
					<img src={reactLogo} className="logo react" alt="React logo" />
				</a>
			</div>
			<h1>Vite + React</h1>

			<div className="audio-recorder-section">
				<h2>Custom Audio Recorder</h2>

				{/* SSE Connection Status */}
				<div className="sse-status">
					<h3>📡 Server Events (SSE)</h3>
					<div className="config-row">
						<span>Connection: </span>
						<span className={sseConnected ? 'connected' : 'disconnected'}>
							{sseConnected ? '✅ Connected' : '❌ Disconnected'}
						</span>
						<span style={{ marginLeft: '20px' }}>Client ID: {clientId}</span>
					</div>
					{lastTimestamp && (
						<div className="config-row">
							<span>🕐 Last Timestamp: {lastTimestamp}</span>
						</div>
					)}
				</div>

				{/* Wake Word Detection Configuration */}
				<div className="chunk-config">
					<div className="config-row">
						<span>🔥 Wake Word Detection: Always Enabled</span>
					</div>

					<div className="config-row">
						<label>
							<input
								type="checkbox"
								checked={streamingMode}
								onChange={(e) => setStreamingMode(e.target.checked)}
								disabled={isRecording}
							/>
							🌊 Continuous Streaming Mode (100ms intervals)
						</label>
					</div>

					{!streamingMode && (
						<div className="config-row">
							<label>
								Chunk Interval:
								<select
									value={chunkInterval}
									onChange={(e) => setChunkInterval(Number(e.target.value))}
									disabled={isRecording}
								>
									<option value={1000}>1 second</option>
									<option value={2000}>2 seconds</option>
									<option value={3000}>3 seconds</option>
									<option value={5000}>5 seconds</option>
									<option value={10000}>10 seconds</option>
								</select>
							</label>
						</div>
					)}
				</div>

				<div className="recorder-controls">
					{!isRecording ? (
						<button
							onClick={startRecording}
							className="record-btn start"
							type="button"
						>
							🎤 Start Recording
						</button>
					) : (
						<div className="recording-controls">
							<button
								onClick={stopRecording}
								className="record-btn stop"
								type="button"
							>
								⏹️ Stop
							</button>

							{!isPaused ? (
								<button
									onClick={pauseRecording}
									className="record-btn pause"
									type="button"
								>
									⏸️ Pause
								</button>
							) : (
								<button
									onClick={resumeRecording}
									className="record-btn resume"
									type="button"
								>
									▶️ Resume
								</button>
							)}
						</div>
					)}
				</div>

				{(isRecording || isUploading) && (
					<div className="recording-info">
						<div className="recording-status">
							<span
								className={`status-indicator ${
									isUploading ? 'uploading' : isPaused ? 'paused' : 'recording'
								}`}
							></span>
							<span>
								{isUploading
									? 'Uploading to server...'
									: isPaused
									? 'Paused'
									: streamingMode
									? '🌊 Streaming'
									: 'Recording'}
								: {isRecording ? formatTime(recordingTime) : ''}
							</span>
						</div>

						{isRecording && (
							<div className="audio-visualizer">
								<div
									className="audio-level-bar"
									style={{ width: `${audioLevelPercentage}%` }}
								></div>
							</div>
						)}
					</div>
				)}

				{/* WebRTC Integration */}
				<WebRTCIntegration
					isRecording={isRecording}
					audioStream={streamRef.current}
				/>

				{/* Wake Word Detection Results */}
				<div className="transcription-results">
					<h3>Wake Word Detection</h3>
					{(wakeWordDetected || serverWakeWordDetected) && (
						<div className="wake-word-alert">🔥 WAKE WORD DETECTED! 🔥</div>
					)}

					{/* Real-time Audio Level from Server */}
					{isRecording && (
						<div className="server-audio-level">
							<h4>📊 Server Audio Level: {realTimeAudioLevel.toFixed(0)}</h4>
							<div className="audio-visualizer">
								<div
									className="audio-level-bar"
									style={{
										width: `${serverAudioLevelPercentage}%`,
									}}
								></div>
							</div>
						</div>
					)}

					{transcriptionResults.length > 0 && (
						<div className="transcription-list">
							{transcriptionResults.map((result, index) => (
								<div key={index} className="transcription-item">
									<span className="chunk-info">
										Chunk {result.chunkIndex} (
										{new Date(result.timestamp).toLocaleTimeString()}):
									</span>
									<span className="transcription-text">{result.text}</span>
								</div>
							))}
						</div>
					)}
				</div>

				{/* SSE Events Debug Info */}
				{sseEvents.length > 0 && (
					<div className="sse-events-info">
						<h4>📡 Recent SSE Events ({sseEvents.length})</h4>
						<div className="sse-events-list">
							{recentSseEvents.map((event, index) => (
								<div key={index} className="sse-event-item">
									<span className="event-type">
										{event.type === 'timestamp_test' && '🕐'}
										{event.type === 'wake_word_detected' && '🔥'}
										{event.type === 'agent_start' && '🤖'}
										{event.type === 'connected' && '✅'}
										{event.type === 'keepalive' && '💓'}
										{event.type}
									</span>
									<span className="event-time">
										{new Date(event.timestamp * 1000).toLocaleTimeString()}
									</span>
									{event.type === 'timestamp_test' && (
										<span className="event-data">
											{(event.data as { datetime?: string }).datetime}
										</span>
									)}
									{event.type === 'wake_word_detected' && (
										<span className="event-data">
											Confidence:{' '}
											{(
												(event.data as { confidence?: number }).confidence || 0
											).toFixed(3)}
										</span>
									)}
									{event.type === 'agent_start' && (
										<span className="event-data">
											{(event.data as { message?: string }).message}{' '}
											(Confidence:{' '}
											{(
												(event.data as { confidence?: number }).confidence || 0
											).toFixed(3)}
											)
										</span>
									)}
								</div>
							))}
						</div>
					</div>
				)}

				{/* Audio Chunks Debug Info */}
				{audioChunks.length > 0 && (
					<div className="chunks-info">
						<h4>Audio Chunks Processed: {audioChunks.length}</h4>
						<div className="chunks-list">
							{recentAudioChunks.map((chunk, index) => (
								<div key={index} className="chunk-item">
									<span>
										Chunk {chunk.chunkIndex}:{' '}
										{(chunk.blob.size / 1024).toFixed(2)} KB
									</span>
									<span className="chunk-time">
										{new Date(chunk.timestamp).toLocaleTimeString()}
									</span>
								</div>
							))}
						</div>
					</div>
				)}

				{recordings.length > 0 && (
					<div className="recordings-list">
						<h3>Recorded Audio Files ({recordings.length})</h3>
						{recordings.map((recording, index) => (
							<div key={index} className="recording-item">
								<div className="recording-info-item">
									<strong>{recording.name}</strong>
									<span className="recording-duration">
										Duration: {formatTime(recording.duration)}
									</span>
									<span className="recording-date">
										{new Date(recording.timestamp).toLocaleString()}
									</span>
								</div>
								<audio controls src={recording.url} />
								<div className="recording-actions">
									<button
										onClick={() => deleteRecording(index)}
										className="delete-btn"
									>
										🗑️ Delete
									</button>
								</div>
							</div>
						))}
					</div>
				)}

				{/* GPS & Maps Integration Demo */}
				<div style={{ marginTop: '40px', borderTop: '2px solid #eee', paddingTop: '20px' }}>
					<GPSMapsDemo />
				</div>
			</div>

			<p className="read-the-docs">
				Click on the Vite and React logos to learn more
			</p>
		</>
	);
}

export default App;
