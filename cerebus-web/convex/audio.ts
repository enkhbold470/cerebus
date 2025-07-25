import { v } from 'convex/values';
import { mutation } from './_generated/server';

export const getAudioUrl = mutation({
	args: {
		storageId: v.id('_storage'),
	},
	handler: async (ctx, args) => {
		return await ctx.storage.getUrl(args.storageId);
	},
});
