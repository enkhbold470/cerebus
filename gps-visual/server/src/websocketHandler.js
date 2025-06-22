const WebSocket = require("ws");
const { getRouteData, getNearbyPlaces, getDistance } = require("./mapsService");

// In-memory store for client states
const clientStates = new Map();

/**
 * Handles incoming WebSocket messages for a given client.
 * @param {WebSocket} ws The WebSocket connection instance.
 * @param {object} parsedMessage The parsed JSON message from the client.
 */
async function handleWebSocketMessage(ws, parsedMessage) {
  const clientState = clientStates.get(ws);

  if (parsedMessage.type === "getRoute") {
    const { origin, destination, destinationName } = parsedMessage.payload;
    if (!origin || !destination) {
      throw new Error("Origin and destination are required");
    }

    console.log("Calculating route for client...");
    const routeData = await getRouteData(origin, destination, destinationName);

    // Store the route and reset navigation state for the client
    clientState.route = routeData;
    clientState.nextStepIndex = 0;

    ws.send(JSON.stringify({ type: "routeResult", payload: routeData }));
    console.log("Route data sent to client.");
  } else if (parsedMessage.type === "searchNearby") {
    const { location, query, radius } = parsedMessage.payload;
    if (!location || !query) {
      throw new Error("Location and query are required for search");
    }

    console.log(`Searching for '${query}' near client...`);
    const placesData = await getNearbyPlaces(location, query, radius);

    ws.send(JSON.stringify({ type: "searchResult", payload: placesData }));
    console.log(`Found ${placesData.length} places. Sent to client.`);
  } else if (parsedMessage.type === "updateGps") {
    const { lat, lng } = parsedMessage.payload;
    if (typeof lat === "number" && typeof lng === "number") {
      clientState.lastGps = { lat, lng };
      checkNavigationProximity(ws, clientState);
    }
  } else if (parsedMessage.type === "wakeWordDetected") {
    console.log("Wake word detected from client.");
    ws.send(
      JSON.stringify({
        type: "audioResponse",
        payload: { text: "How can I help you?" },
      })
    );
  }
}

/**
 * Checks if the client is close to the next navigation step and sends an instruction if they are.
 * @param {WebSocket} ws The WebSocket connection instance.
 * @param {object} clientState The state object for the client.
 */
function checkNavigationProximity(ws, clientState) {
  if (
    clientState.route &&
    clientState.nextStepIndex < clientState.route.detailed_steps.length
  ) {
    const nextStep =
      clientState.route.detailed_steps[clientState.nextStepIndex];
    const distanceToNextStep = getDistance(
      clientState.lastGps,
      nextStep.start_location
    );

    const PROXIMITY_THRESHOLD = 20; // in meters

    if (distanceToNextStep <= PROXIMITY_THRESHOLD) {
      const instructionText = nextStep.instruction.replace(/<[^>]*>/g, "");
      console.log(
        `Client is close to next step. Sending instruction: "${instructionText}"`
      );
      ws.send(
        JSON.stringify({
          type: "audioResponse",
          payload: { text: instructionText },
        })
      );
      clientState.nextStepIndex++;
    }
  }
}

/**
 * Initializes the WebSocket server and sets up connection handlers.
 * @param {http.Server} server The HTTP server to attach the WebSocket server to.
 */
function initializeWebSocket(server) {
  const wss = new WebSocket.Server({ server });

  wss.on("connection", (ws) => {
    console.log("Client connected to WebSocket");
    clientStates.set(ws, { lastGps: null, route: null, nextStepIndex: 0 });

    ws.on("message", async (message) => {
      try {
        const parsedMessage = JSON.parse(message);
        await handleWebSocketMessage(ws, parsedMessage);
      } catch (error) {
        console.error("Error processing message:", error.message);
        ws.send(
          JSON.stringify({
            type: "error",
            payload: { message: error.message },
          })
        );
      }
    });

    ws.on("close", () => {
      console.log("Client disconnected");
      clientStates.delete(ws);
    });
  });
}

module.exports = { initializeWebSocket };
