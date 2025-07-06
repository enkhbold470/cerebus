require("dotenv").config({ path: "../.env" });
const express = require("express");
const http = require("http");
const path = require("path");
const cors = require("cors");
const apiRoutes = require("./src/apiRoutes");
const { initializeWebSocket } = require("./src/websocketHandler");

const app = express();
const port = process.env.PORT || 3000;

// --- Middleware ---
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "..")));

// --- API Routes ---
app.use("/api", apiRoutes);

// --- Server Initialization ---
const server = http.createServer(app);

// --- WebSocket Initialization ---
initializeWebSocket(server);

// --- Start Server ---
server.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
  console.log("WebSocket server is ready for connections.");
});
