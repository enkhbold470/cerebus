# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default tseslint.config({
	extends: [
		// Remove ...tseslint.configs.recommended and replace with this
		...tseslint.configs.recommendedTypeChecked,
		// Alternatively, use this for stricter rules
		...tseslint.configs.strictTypeChecked,
		// Optionally, add this for stylistic rules
		...tseslint.configs.stylisticTypeChecked,
	],
	languageOptions: {
		// other options...
		parserOptions: {
			project: ['./tsconfig.node.json', './tsconfig.app.json'],
			tsconfigRootDir: import.meta.dirname,
		},
	},
});
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x';
import reactDom from 'eslint-plugin-react-dom';

export default tseslint.config({
	plugins: {
		// Add the react-x and react-dom plugins
		'react-x': reactX,
		'react-dom': reactDom,
	},
	rules: {
		// other rules...
		// Enable its recommended typescript rules
		...reactX.configs['recommended-typescript'].rules,
		...reactDom.configs.recommended.rules,
	},
});
```

# Cerebus Web - Audio Recording with Wake Word Detection

This project combines a React + Vite frontend with a Convex backend for audio recording and real-time wake word detection.

## Features

- **Audio Recording**: Record, pause, resume, and stop audio with real-time visualization
- **Local Download**: Automatic local download of recordings
- **Cloud Storage**: Upload recordings to Convex for persistence
- **Wake Word Detection**: Real-time audio chunk processing for wake word detection
- **Real-time Feedback**: Live audio level monitoring and recording status

## Setup

### 1. Install Dependencies

```bash
pnpm install
```

### 2. Configure Convex

```bash
npx convex dev
```

### 3. Wake Word Detection Server

To enable wake word detection, you'll need to run the Python FastAPI server:

1. **Install Python Dependencies**:

   ```bash
   pip install fastapi uvicorn openwakeword numpy subprocess pathlib tempfile
   ```

2. **Install FFmpeg** (required for audio conversion):

   - **macOS**: `brew install ffmpeg`
   - **Ubuntu**: `sudo apt update && sudo apt install ffmpeg`
   - **Windows**: Download from https://ffmpeg.org/download.html

3. **Place your Wake Word Model**:

   - Put your `hey_cerebus.onnx` model file in the same directory as the server script
   - Or modify the `--model_path` argument to point to your model

4. **Run the Wake Word Server**:

   ```bash
   python wake_word_server.py
   ```

   The server will start on `http://localhost:8000` by default.

### 4. Start the Frontend

```bash
pnpm run dev
```

## Usage

1. **Basic Recording**: Click "Start Recording" to record audio
2. **Wake Word Detection**:
   - Check "Enable Wake Word Detection"
   - Set your preferred chunk interval (1-10 seconds)
   - Start recording to begin real-time wake word detection
3. **Results**: Wake word detections will appear in real-time with visual alerts

## Configuration

- **Wake Word Server URL**: Update `WAKE_WORD_SERVER_URL` in `constants.ts`
- **Chunk Intervals**: Configure real-time processing intervals in the UI
- **Audio Settings**: Modify audio constraints in `App.tsx`

## Architecture

- **Frontend**: React + TypeScript + Vite
- **Backend**: Convex (serverless functions + file storage)
- **Wake Word Detection**: Python FastAPI server with OpenWakeWord
- **Audio Processing**: MediaRecorder API with real-time chunking

The frontend sends audio chunks directly to your wake word detection server for real-time processing.
