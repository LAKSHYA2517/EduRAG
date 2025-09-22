// routes/api.js
const express = require('express');
const router = express.Router();
const uploadMiddleware = require('../middleware/uploadMiddleware');
const { uploadDocument, askQuestion } = require('../controllers/qaController');

// Endpoint for uploading the document and creating a session
// POST /api/upload
router.post('/upload', uploadMiddleware, uploadDocument);

// Endpoint for asking a question about the uploaded document
// POST /api/ask
router.post('/ask', askQuestion); // No middleware needed here

module.exports = router;