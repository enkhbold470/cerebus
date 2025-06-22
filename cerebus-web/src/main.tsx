import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import { ConvexProvider, ConvexReactClient } from 'convex/react';

import App from './App.tsx';
const convex = new ConvexReactClient(
	'https://blessed-reindeer-300.convex.cloud',
);

createRoot(document.getElementById('root')!).render(
	<StrictMode>
		<ConvexProvider client={convex}>
			<App />
		</ConvexProvider>
	</StrictMode>,
);
