// backend/controllers/qaController.js

const axios = require('axios');
require('dotenv').config();

// Assumes your teammate's ML API has these two endpoints
const ML_UPLOAD_URL = process.env.ML_API_BASE_URL + '/load-document';
const ML_QUERY_URL = process.env.ML_API_BASE_URL + '/query';

// Function to handle the initial document upload
const uploadDocument = async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ message: 'No document file uploaded.' });
    }
    try {
        const mlResponse = await axios.post(ML_UPLOAD_URL, {
            filePath: req.file.path
        });
        if (!mlResponse.data.sessionId) {
            throw new Error('ML service did not return a sessionId.');
        }
        res.status(200).json({
            message: 'Document processed successfully.',
            sessionId: mlResponse.data.sessionId
        });
    } catch (error) {
        console.error('Error during document upload:', error.message);
        res.status(500).json({ message: 'Failed to process document.' });
    }
};

// Function to handle a user's question
const askQuestion = async (req, res) => {
    const { sessionId, question } = req.body;
    if (!sessionId || !question) {
        return res.status(400).json({ message: 'sessionId and question are required.' });
    }
    try {
        const mlResponse = await axios.post(ML_QUERY_URL, {
            sessionId: sessionId,
            question: question
        });
        if (!mlResponse.data.answer) {
            throw new Error('ML service did not return an answer.');
        }
        res.status(200).json({
            answer: mlResponse.data.answer
        });
    } catch (error) {
        console.error('Error during question asking:', error.message);
        res.status(500).json({ message: 'Failed to get an answer from the service.' });
    }
};

module.exports = {
    uploadDocument,
    askQuestion
};