const mongoose = require('mongoose');

const textbookSchema = new mongoose.Schema({
    title: {
        type: String,
        required: true
    },
    author: {
        type: String
    },
    subject: {
        type: String,
        required: true
    },
    // Path to the textbook PDF on your server's filesystem
    filePath: {
        type: String,
        required: true
    }
});

module.exports = mongoose.model('Textbook', textbookSchema);
