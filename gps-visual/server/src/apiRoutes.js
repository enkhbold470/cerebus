const express = require("express");
const multer = require("multer");
const { processImage } = require("./imageProcessor");
const { analyzeWithGemini, analyzeWithGroq } = require("./aiService");
const { getRouteData, getNearbyPlaces } = require("./mapsService");

const router = express.Router();

// Multer setup for in-memory file storage
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

/**
 * @api {post} /api/analyze-image Analyze Image with Gemini
 * @apiName AnalyzeImageGemini
 * @apiGroup Vision
 *
 * @apiDescription This endpoint accepts an image file, processes it,
 * and returns a JSON analysis from Google's Gemini model.
 *
 * @apiParam {File} image The image file to analyze.
 *
 * @apiSuccess {Object} analysis The JSON analysis of the image.
 * @apiError {String} error Description of the error.
 */
router.post("/analyze-image", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No image file uploaded." });
  }

  try {
    const processedBuffer = await processImage(req.file.buffer);
    const analysisResult = await analyzeWithGemini(processedBuffer);
    res.json(analysisResult);
  } catch (error) {
    console.error("Error with Gemini API:", error);
    res.status(500).json({ error: "Failed to analyze the image." });
  }
});

/**
 * @api {post} /api/analyze-image-groq Analyze Image with Groq
 * @apiName AnalyzeImageGroq
 * @apiGroup Vision
 *
 * @apiDescription This endpoint accepts an image file, processes it,
 * and returns a JSON analysis from the Groq service.
 *
 * @apiParam {File} image The image file to analyze.
 *
 * @apiSuccess {Object} analysis The JSON analysis of the image.
 * @apiError {String} error Description of the error.
 */
router.post(
  "/api/analyze-image-groq",
  upload.single("image"),
  async (req, res) => {
    if (!req.file) {
      return res.status(400).json({ error: "No image file uploaded." });
    }

    try {
      const processedBuffer = await processImage(req.file.buffer);
      const analysisResult = await analyzeWithGroq(processedBuffer);
      res.json(analysisResult);
    } catch (error) {
      console.error("Error with Groq API:", error);
      res.status(500).json({ error: "Failed to analyze the image with Groq." });
    }
  }
);

router.post("/directions", async (req, res) => {
  const { origin, destination, destinationName } = req.body;
  if (!origin || !destination) {
    return res
      .status(400)
      .json({ error: "Origin and destination are required." });
  }

  try {
    const routeData = await getRouteData(origin, destination, destinationName);
    res.json(routeData);
  } catch (error) {
    console.error("Error with Directions API:", error);
    res.status(500).json({ error: "Failed to get directions." });
  }
});

router.post("/nearby-places", async (req, res) => {
  const { location, query, radius } = req.body;
  if (!location || !query) {
    return res.status(400).json({ error: "Location and query are required." });
  }

  try {
    const places = await getNearbyPlaces(location, query, radius);
    res.json(places);
  } catch (error) {
    console.error("Error with Places API:", error);
    res.status(500).json({ error: "Failed to get nearby places." });
  }
});

module.exports = router;
