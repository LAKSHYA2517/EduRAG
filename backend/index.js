const express = require('express');
const cors = require('cors');
require('dotenv').config();
const connectDB = require('./config/db');

const app = express();

// Connect to Database
connectDB();

// Middlewares
app.use(cors()); // Allow cross-origin requests
app.use(express.json()); // To parse JSON bodies
app.use(express.urlencoded({ extended: false }));

// Define Routes
app.use('/api', require('./routes/api'));

// Simple base route for testing
app.get('/', (req, res) => {
    res.send('Backend API is running...');
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => console.log(`🚀 Server started on port ${PORT}`));