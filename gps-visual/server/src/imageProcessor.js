const sharp = require("sharp");

/**
 * Processes and compresses an image buffer.
 * @param {Buffer} imageBuffer The raw image buffer from the upload.
 * @returns {Promise<Buffer>} A promise that resolves with the compressed JPEG image buffer.
 */
async function processImage(imageBuffer) {
  return sharp(imageBuffer)
    .resize({ width: 800, height: 600, fit: "inside" })
    .jpeg({ quality: 75 })
    .toBuffer();
}

module.exports = {
  processImage,
};
