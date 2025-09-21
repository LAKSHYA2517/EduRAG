const Syllabus = require('../models/syllabusModel');
const axios = require('axios'); // To call the ML service
require('dotenv').config();

const getPredictedQuestions = async (req, res) => {
    // 1. Check if a file was uploaded
    if (!req.file) {
        return res.status(400).json({ message: 'No syllabus file uploaded.' });
    }

    try {
        // 2. Save file metadata to our MongoDB database
        const newSyllabus = new Syllabus({
            originalName: req.file.originalname,
            filePath: req.file.path // The path where Multer saved the file
        });
        await newSyllabus.save();

        console.log(`File metadata saved. Path: ${req.file.path}`);
        console.log(`Calling ML service at: ${process.env.ML_API_URL}`);

        // 3. Trigger the ML RAG Pipeline
        // We make a POST request to the ML service API and send the path of the uploaded file.
        const mlResponse = await axios.post(process.env.ML_API_URL, {
            syllabus_path: req.file.path
        });

        // 4. Check if the ML service responded successfully
        if (mlResponse.status !== 200 || !mlResponse.data.predicted_questions) {
            throw new Error('Invalid response from ML service');
        }
        
        // 5. Send the predicted questions back to the frontend
        res.status(200).json({
            message: 'Questions generated successfully!',
            questions: mlResponse.data.predicted_questions
        });

    } catch (error) {
        console.error('Error during prediction process:', error.message);
        // More detailed error logging for debugging
        if (error.response) {
            console.error('ML Service Response Error:', error.response.data);
        }
        res.status(500).json({ message: 'Server error during question generation.' });
    }
};

module.exports = {
    getPredictedQuestions
};