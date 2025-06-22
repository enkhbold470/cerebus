import { defineSchema, defineTable } from 'convex/server';
import { v } from 'convex/values';

export default defineSchema({
	audioRecords: defineTable({
		storageId: v.id('_storage'),
		contentType: v.string(),
		size: v.number(),
		uploadedAt: v.number(),
		filename: v.optional(v.string()),
		duration: v.optional(v.number()), // Duration in seconds
		transcription: v.optional(v.string()), // For future transcription storage
		tags: v.optional(v.array(v.string())), // For categorization
	})
		.index('by_upload_date', ['uploadedAt'])
		.index('by_content_type', ['contentType']),

	audioChunks: defineTable({
		storageId: v.id('_storage'),
		contentType: v.string(),
		size: v.number(),
		timestamp: v.number(), // When the chunk was recorded
		chunkIndex: v.number(), // Sequential chunk number
		fileName: v.string(), // Original filename of the chunk
		transcription: v.optional(v.string()), // Transcribed text
		processedAt: v.number(), // When the chunk was processed
	})
		.index('by_timestamp', ['timestamp'])
		.index('by_chunk_index', ['chunkIndex']),
});
