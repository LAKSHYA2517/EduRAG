const mongoose = require('mongoose');

const pyqSchema = new mongoose.Schema({
    subject: {
        type: String,
        required: true,
        trim: true
    },
    courseCode: {
        type: String,
        unique: true,
        trim: true
    },
    // Path to the combined PDF file on your server's filesystem
    filePath: {
        type: String,
        required: true
    },
    uploadedAt: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model('PYQ', pyqSchema);