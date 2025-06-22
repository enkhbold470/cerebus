import { httpRouter } from 'convex/server';
import { api } from './_generated/api';
import { httpAction } from './_generated/server';

const http = httpRouter();

// HTTP endpoint to handle audio chunks for real-time processing
http.route({
	path: '/audio/chunk',
	method: 'POST',
	handler: httpAction(async (ctx, request) => {
		try {
			console.log('Processing audio chunk');

			// Parse the FormData from the request
			const formData = await request.formData();

			// Extract the data from FormData
			const audioFile = formData.get('audio') as File;
			const timestamp = formData.get('timestamp') as string;
			const chunkIndex = formData.get('chunkIndex') as string;

			if (!audioFile) {
				return new Response(
					JSON.stringify({
						error: 'No audio file received in chunk',
					}),
					{
						status: 400,
						headers: { 'Content-Type': 'application/json' },
					},
				);
			}

			// Convert file to ArrayBuffer
			const audioBuffer = await audioFile.arrayBuffer();
			const contentType = 'audio/webm';

			console.log(
				`Processing chunk ${chunkIndex} - Size: ${audioBuffer.byteLength} bytes`,
			);

			// Store the audio chunk in Convex file storage
			const audioBlob = new Blob([audioBuffer], { type: contentType });
			const storageId = await ctx.storage.store(audioBlob);

			// Save chunk metadata to database
			const chunkId = await ctx.runMutation(api.audioMutations.saveAudioChunk, {
				storageId,
				contentType,
				size: audioBuffer.byteLength,
				timestamp: parseInt(timestamp) || Date.now(),
				chunkIndex: parseInt(chunkIndex) || 0,
				fileName: audioFile.name,
			});

			// Placeholder for transcription logic (not used for wake word detection)
			const transcriptionText = `[Chunk ${chunkIndex} processed at ${new Date().toISOString()}]`;

			return new Response(
				JSON.stringify({
					success: true,
					chunkId,
					storageId,
					chunkIndex: parseInt(chunkIndex),
					timestamp: parseInt(timestamp),
					size: audioBuffer.byteLength,
					contentType,
					text: transcriptionText, // Transcribed text
					message: 'Audio chunk processed successfully',
				}),
				{
					status: 200,
					headers: {
						'Content-Type': 'application/json',
						'Access-Control-Allow-Origin': '*',
						'Access-Control-Allow-Methods': 'POST, OPTIONS',
						'Access-Control-Allow-Headers': 'Content-Type',
					},
				},
			);
		} catch (error) {
			console.error('Audio chunk processing error:', error);
			return new Response(
				JSON.stringify({
					error: 'Failed to process audio chunk',
					details: error instanceof Error ? error.message : 'Unknown error',
				}),
				{
					status: 500,
					headers: { 'Content-Type': 'application/json' },
				},
			);
		}
	}),
});

// Handle CORS preflight requests for chunk endpoint
http.route({
	path: '/audio/chunk',
	method: 'OPTIONS',
	handler: httpAction(async () => {
		return new Response(null, {
			status: 200,
			headers: {
				'Access-Control-Allow-Origin': '*',
				'Access-Control-Allow-Methods': 'POST, OPTIONS',
				'Access-Control-Allow-Headers': 'Content-Type',
			},
		});
	}),
});

// Note: Wake word detection now happens directly from frontend to Python server

// HTTP endpoint to receive audio data
http.route({
	path: '/audio/upload',
	method: 'POST',
	handler: httpAction(async (ctx, request) => {
		try {
			console.log('Uploading audio');
			// Check if the request has audio content

			// Get the audio data as ArrayBuffer
			const audioBuffer = await request.arrayBuffer();
			const contentType = request.headers.get('content-type') || 'audio/m4a';

			if (audioBuffer.byteLength === 0) {
				console.log('No audio data received.');
				return new Response(
					JSON.stringify({
						error: 'No audio data received.',
					}),
					{
						status: 400,
						headers: { 'Content-Type': 'application/json' },
					},
				);
			}

			// Store the audio file in Convex file storage
			const audioBlob = new Blob([audioBuffer], { type: contentType });
			const storageId = await ctx.storage.store(audioBlob);

			console.log('Storage ID:', storageId);

			// Save audio metadata to database
			const audioId = await ctx.runMutation(
				api.audioMutations.saveAudioRecord,
				{
					storageId,
					contentType,
					size: audioBuffer.byteLength,
					uploadedAt: Date.now(),
				},
			);

			return new Response(
				JSON.stringify({
					success: true,
					audioId,
					storageId,
					size: audioBuffer.byteLength,
					contentType,
					message: 'Audio uploaded successfully',
				}),
				{
					status: 200,
					headers: {
						'Content-Type': 'application/json',
						'Access-Control-Allow-Origin': '*',
						'Access-Control-Allow-Methods': 'POST, OPTIONS',
						'Access-Control-Allow-Headers': 'Content-Type',
					},
				},
			);
		} catch (error) {
			console.error('Audio upload error:', error);
			return new Response(
				JSON.stringify({
					error: 'Failed to upload audio',
					details: error instanceof Error ? error.message : 'Unknown error',
				}),
				{
					status: 500,
					headers: { 'Content-Type': 'application/json' },
				},
			);
		}
	}),
});

// Handle CORS preflight requests
http.route({
	path: '/audio/upload',
	method: 'OPTIONS',
	handler: httpAction(async () => {
		return new Response(null, {
			status: 200,
			headers: {
				'Access-Control-Allow-Origin': '*',
				'Access-Control-Allow-Methods': 'POST, OPTIONS',
				'Access-Control-Allow-Headers': 'Content-Type',
			},
		});
	}),
});

export default http;
