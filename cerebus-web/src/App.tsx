import { useState, useRef, useEffect } from 'react';
import reactLogo from './assets/react.svg';
import viteLogo from '/vite.svg';
import { api } from '../convex/_generated/api';
import { WAKE_WORD_SERVER_URL } from '../constants';

import './App.css';
import { useMutation } from 'convex/react';

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

	const mediaRecorderRef = useRef<MediaRecorder | null>(null);
	const audioChunksRef = useRef<BlobPart[]>([]);
	const streamRef = useRef<MediaStream | null>(null);
	const timerRef = useRef<number | null>(null);
	const analyserRef = useRef<AnalyserNode | null>(null);
	const animationFrameRef = useRef<number | null>(null);

	// Cleanup function
	const cleanupRecording = () => {
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
	};

	// Audio level monitoring for visual feedback
	const monitorAudioLevel = () => {
		if (!analyserRef.current) return;

		const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
		analyserRef.current.getByteFrequencyData(dataArray);

		const average =
			dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
		setAudioLevel(average);

		if (isRecording && !isPaused) {
			animationFrameRef.current = requestAnimationFrame(monitorAudioLevel);
		}
	};

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

				// Upload to Convex
				setIsUploading(true);
				try {
					await uploadAudioToConvex(audioBlob);
					console.log('Recording uploaded to Convex successfully');
				} catch (error) {
					console.error('Failed to upload to Convex:', error);
				} finally {
					setIsUploading(false);
				}

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
	const getUploadUrl = useMutation(api.audioMutations.getUploadUrl);

	const uploadAudioToConvex = async (audioBlob: Blob) => {
		try {
			// Get upload URL from Convex
			const uploadUrl = await getUploadUrl();

			// Upload the blob to Convex
			const response = await fetch(uploadUrl, {
				method: 'POST',
				body: audioBlob,
			});

			if (!response.ok) {
				throw new Error(`Upload failed: ${response.statusText}`);
			}

			const result = await response.json();
			console.log('Upload successful:', result);
			return result;
		} catch (error) {
			console.error('Error uploading audio to Convex:', error);
			alert(
				'Failed to upload recording to server. The file was still saved locally.',
			);
			throw error;
		}
	};

	const uploadChunkForTranscription = async (chunk: AudioChunk) => {
		try {
			// Send directly to the wake word detection server
			const wakeWordEndpoint = `${WAKE_WORD_SERVER_URL}/detect`;

			console.log(
				`🚀 Sending chunk ${chunk.chunkIndex} to wake word server: ${wakeWordEndpoint}`,
			);

			const formData = new FormData();
			formData.append('audio', chunk.blob, `chunk_${chunk.chunkIndex}.webm`);

			const response = await fetch(wakeWordEndpoint, {
				method: 'POST',
				body: formData,
			});

			if (!response.ok) {
				throw new Error(`Wake word detection failed: ${response.statusText}`);
			}

			const result = await response.json();

			// Process wake word detection results
			let displayText = '🔍 Listening...';
			let wakeWordFound = false;

			if (result.detections && result.detections.length > 0) {
				const detections = result.detections
					.map(
						(detection: { model: string; score: number }) =>
							`${detection.model} (${(detection.score * 100).toFixed(1)}%)`,
					)
					.join(', ');
				displayText = `🔥 WAKE WORD DETECTED: ${detections}`;
				wakeWordFound = true;
			}

			// Add result to state
			const transcriptionResult: TranscriptionResult = {
				text: displayText,
				timestamp: chunk.timestamp,
				chunkIndex: chunk.chunkIndex,
			};

			setTranscriptionResults((prev) => [...prev, transcriptionResult]);

			// Update wake word detection state
			if (wakeWordFound) {
				setWakeWordDetected(true);
				// Reset after 5 seconds
				setTimeout(() => setWakeWordDetected(false), 5000);
			}

			return result;
		} catch (error) {
			console.error('Error transcribing audio chunk:', error);
			throw error;
		}
	};

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

			// Upload chunk for wake word detection
			try {
				await uploadChunkForTranscription(chunk);
			} catch (error) {
				console.error('Failed to process chunk:', error);
			}
		}
	};

	const stopRecording = () => {
		if (
			mediaRecorderRef.current &&
			mediaRecorderRef.current.state !== 'inactive'
		) {
			mediaRecorderRef.current.stop();
			setIsRecording(false);
		}
	};

	const pauseRecording = () => {
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
	};

	const resumeRecording = () => {
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
	};

	const deleteRecording = (index: number) => {
		setRecordings((prev) => {
			const updated = [...prev];
			URL.revokeObjectURL(updated[index].url); // Clean up memory
			updated.splice(index, 1);
			return updated;
		});
	};

	// Format time display
	const formatTime = (seconds: number) => {
		const mins = Math.floor(seconds / 60);
		const secs = seconds % 60;
		return `${mins.toString().padStart(2, '0')}:${secs
			.toString()
			.padStart(2, '0')}`;
	};

	// Cleanup on component unmount
	useEffect(() => {
		return () => {
			cleanupRecording();
		};
	}, []);

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
									style={{ width: `${(audioLevel / 255) * 100}%` }}
								></div>
							</div>
						)}

						{isUploading && (
							<div className="upload-status">
								<span>📤 Uploading recording to Convex...</span>
							</div>
						)}
					</div>
				)}

				{/* Wake Word Detection Results */}
				<div className="transcription-results">
					<h3>Wake Word Detection</h3>
					{wakeWordDetected && (
						<div className="wake-word-alert">🔥 WAKE WORD DETECTED! 🔥</div>
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

				{/* Audio Chunks Debug Info */}
				{audioChunks.length > 0 && (
					<div className="chunks-info">
						<h4>Audio Chunks Processed: {audioChunks.length}</h4>
						<div className="chunks-list">
							{audioChunks.slice(-5).map((chunk, index) => (
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
			</div>

			<p className="read-the-docs">
				Click on the Vite and React logos to learn more
			</p>
		</>
	);
}

export default App;
