const { GoogleGenerativeAI } = require("@google/generative-ai");
const Groq = require("groq-sdk");

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const groqClient = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

const VISION_PROMPT = `You are an AI assistant for a visually impaired user. Your purpose is to be their "eyes."
This image is a snapshot from the smart glasses they are wearing.
Analyze the image from their point of view and provide a description in JSON format.
The JSON should have the following structure:
{
  "description": "A brief, one-sentence summary of what is happening in the scene.",
  "immediate_obstacles": ["A list of any immediate obstacles or hazards in the user's direct path, like 'curb', 'puddle', 'low-hanging branch', 'oncoming bicycle'. If none, the array should be empty."],
  "navigation_suggestion": "If an immediate obstacle is detected, provide a clear, verbal command to navigate around it safely. Example: 'Obstacle detected. Take two steps left, then proceed forward.' If no obstacle, this field should be an empty string.",
  "points_of_interest": ["A list of notable objects or landmarks, like 'crosswalk button to the right', 'store entrance ahead', 'park bench on the left'. Keep it concise."],
  "ambient_context": "Describe the general environment, like 'busy city sidewalk', 'quiet park path', 'indoor hallway'."
}
Your primary goal is user safety. The navigation_suggestion is the most important part if an obstacle is present. Be direct and actionable.`;

/**
 * Analyzes an image with Google Gemini.
 * @param {Buffer} imageBuffer The compressed image buffer.
 * @returns {Promise<object>} A promise that resolves with the parsed JSON response from the AI.
 */
async function analyzeWithGemini(imageBuffer) {
  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
  const imagePart = {
    inlineData: {
      data: imageBuffer.toString("base64"),
      mimeType: "image/jpeg",
    },
  };
  const result = await model.generateContent([VISION_PROMPT, imagePart]);
  const responseText = result.response.text();
  const cleanedJsonString = responseText.replace(/```json|```/g, "").trim();
  return JSON.parse(cleanedJsonString);
}

/**
 * Analyzes an image with Groq.
 * @param {Buffer} imageBuffer The compressed image buffer.
 * @returns {Promise<object>} A promise that resolves with the parsed JSON response from the AI.
 */
async function analyzeWithGroq(imageBuffer) {
  const base64Image = imageBuffer.toString("base64");
  const completion = await groqClient.chat.completions.create({
    model: "meta-llama/llama-4-scout-17b-16e-instruct",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: VISION_PROMPT },
          {
            type: "image_url",
            image_url: { url: `data:image/jpeg;base64,${base64Image}` },
          },
        ],
      },
    ],
    temperature: 1,
    max_completion_tokens: 1024,
    top_p: 1,
    stream: false,
    response_format: { type: "json_object" },
    stop: null,
  });

  return JSON.parse(completion.choices[0].message.content);
}

module.exports = {
  analyzeWithGemini,
  analyzeWithGroq,
};
