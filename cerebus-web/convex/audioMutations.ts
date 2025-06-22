import { v } from 'convex/values';
import type { Id } from './_generated/dataModel';
import { mutation, query } from './_generated/server';

// Save audio record to database
export const saveAudioRecord = mutation({
	args: {
		storageId: v.id('_storage'),
		contentType: v.string(),
		size: v.number(),
		uploadedAt: v.number(),
		filename: v.optional(v.string()),
		duration: v.optional(v.number()),
		tags: v.optional(v.array(v.string())),
	},
	handler: async (ctx, args) => {
		const audioId = await ctx.db.insert('audioRecords', {
			storageId: args.storageId,
			contentType: args.contentType,
			size: args.size,
			uploadedAt: args.uploadedAt,
			filename: args.filename,
			duration: args.duration,
			tags: args.tags,
		});
		return audioId;
	},
});

// Save audio chunk to database for real-time processing
export const saveAudioChunk = mutation({
	args: {
		storageId: v.id('_storage'),
		contentType: v.string(),
		size: v.number(),
		timestamp: v.number(),
		chunkIndex: v.number(),
		fileName: v.string(),
		transcription: v.optional(v.string()),
	},
	handler: async (ctx, args) => {
		const chunkId = await ctx.db.insert('audioChunks', {
			storageId: args.storageId,
			contentType: args.contentType,
			size: args.size,
			timestamp: args.timestamp,
			chunkIndex: args.chunkIndex,
			fileName: args.fileName,
			transcription: args.transcription,
			processedAt: Date.now(),
		});
		return chunkId;
	},
});

export const getUploadUrl = mutation({
	handler: async (ctx) => {
		return await ctx.storage.generateUploadUrl();
	},
});

// Get audio record by ID
export const getAudioRecord = mutation({
	args: {
		audioId: v.id('audioRecords'),
	},
	handler: async (ctx, args) => {
		const audioRecord = await ctx.db.get(args.audioId);
		const audioUrl = await ctx.storage.getUrl(
			audioRecord?.storageId as Id<'_storage'>,
		);
		return { ...audioRecord, audioUrl };
	},
});

// List all audio records with pagination and sorting
export const listAudioRecords = query({
	args: {
		limit: v.optional(v.number()),
		orderBy: v.optional(v.union(v.literal('newest'), v.literal('oldest'))),
	},
	handler: async (ctx, args) => {
		const limit = args.limit ?? 50;

		// Apply ordering based on upload date
		if (args.orderBy === 'oldest') {
			return await ctx.db
				.query('audioRecords')
				.withIndex('by_upload_date')
				.order('asc')
				.take(limit);
		} else {
			// Default to newest first
			return await ctx.db
				.query('audioRecords')
				.withIndex('by_upload_date')
				.order('desc')
				.take(limit);
		}
	},
});

// Delete audio record and file
export const deleteAudioRecord = mutation({
	args: {
		audioId: v.id('audioRecords'),
	},
	handler: async (ctx, args) => {
		const audioRecord = await ctx.db.get(args.audioId);
		if (!audioRecord) {
			throw new Error('Audio record not found');
		}

		// Delete the file from storage
		await ctx.storage.delete(audioRecord.storageId);

		// Delete the record from database
		await ctx.db.delete(args.audioId);

		return { success: true };
	},
});

// List audio records by content type
export const listAudioRecordsByType = query({
	args: {
		contentType: v.string(),
		limit: v.optional(v.number()),
	},
	handler: async (ctx, args) => {
		const limit = args.limit ?? 50;
		return await ctx.db
			.query('audioRecords')
			.withIndex('by_content_type', (q) =>
				q.eq('contentType', args.contentType),
			)
			.order('desc')
			.take(limit);
	},
});

// Search audio records by tags
export const searchAudioRecordsByTags = query({
	args: {
		tags: v.array(v.string()),
		limit: v.optional(v.number()),
	},
	handler: async (ctx, args) => {
		const limit = args.limit ?? 50;
		const allRecords = await ctx.db.query('audioRecords').collect();

		// Filter records that contain any of the specified tags
		const filteredRecords = allRecords.filter((record) => {
			if (!record.tags) return false;
			return args.tags.some((tag) => record.tags!.includes(tag));
		});

		return filteredRecords
			.sort((a, b) => b.uploadedAt - a.uploadedAt)
			.slice(0, limit);
	},
});

// Get audio records statistics
export const getAudioStats = query({
	args: {},
	handler: async (ctx) => {
		const records = await ctx.db.query('audioRecords').collect();
		const totalSize = records.reduce((sum, record) => sum + record.size, 0);
		const totalDuration = records.reduce(
			(sum, record) => sum + (record.duration || 0),
			0,
		);

		const contentTypes = records.reduce((acc, record) => {
			acc[record.contentType] = (acc[record.contentType] || 0) + 1;
			return acc;
		}, {} as Record<string, number>);

		return {
			totalRecords: records.length,
			totalSize,
			totalDuration,
			contentTypes,
			averageSize: records.length > 0 ? totalSize / records.length : 0,
		};
	},
});

// Update audio record metadata
export const updateAudioRecord = mutation({
	args: {
		audioId: v.id('audioRecords'),
		filename: v.optional(v.string()),
		duration: v.optional(v.number()),
		transcription: v.optional(v.string()),
		tags: v.optional(v.array(v.string())),
	},
	handler: async (ctx, args) => {
		const audioRecord = await ctx.db.get(args.audioId);
		if (!audioRecord) {
			throw new Error('Audio record not found');
		}

		await ctx.db.patch(args.audioId, {
			filename: args.filename ?? audioRecord.filename,
			duration: args.duration ?? audioRecord.duration,
			transcription: args.transcription ?? audioRecord.transcription,
			tags: args.tags ?? audioRecord.tags,
		});

		return { success: true };
	},
});
