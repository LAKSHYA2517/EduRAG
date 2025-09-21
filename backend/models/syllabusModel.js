const mongoose = require('mongoose');

const syllabusSchema = new mongoose.Schema({
    originalName: {
        type: String,
        required: true
    },
    // Path where the uploaded file is stored on the server
    filePath: {
        type: String,
        required: true
    },
    // You can add a user ID here later for authentication
    // userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model('Syllabus', syllabusSchema);