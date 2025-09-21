const express = require('express');
const router = express.Router();

const uploadMiddleware = require('../middleware/uploadMiddleware');
const { getPredictedQuestions } = require('../controllers/predictController');

// @route   POST /api/predict
// @desc    Upload syllabus and get predicted questions
// @access  Public
router.post('/predict', uploadMiddleware, getPredictedQuestions);

module.exports = router;