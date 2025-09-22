// components/QaContainer.js
import React, { useState } from 'react';
import axios from 'axios';
import DocumentUpload from './DocumentUpload';
import ChatBox from './ChatBox'; // ✅ Import your ChatBox component

const QaContainer = () => {
  const [sessionId, setSessionId] = useState('');
  const [question, setQuestion] = useState(''); // This state will now control the ChatBox input
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleUpload = async (file) => {
    // ... (this function remains exactly the same as before)
    if (!file) return;
    setIsLoading(true);
    setError('');
    const formData = new FormData();
    formData.append('document', file);
    try {
        const response = await axios.post('http://localhost:5000/api/upload', formData);
        setSessionId(response.data.sessionId);
        setChatHistory([{ type: 'system', text: `Ready! Ask questions about ${file.name}.` }]);
    } catch (err) {
        setError(err.response ? err.response.data.message : 'Upload failed.');
    } finally {
        setIsLoading(false);
    }
  };

  const handleAskQuestion = async () => {
    // ... (this function is mostly the same, just no more e.preventDefault())
    if (!question.trim() || !sessionId) return;
    const userQuestion = question;
    setChatHistory((prev) => [...prev, { type: 'user', text: userQuestion }]);
    setQuestion('');
    setIsLoading(true);
    try {
        const response = await axios.post('http://localhost:5000/api/ask', {
            sessionId: sessionId,
            question: userQuestion,
        });
        setChatHistory((prev) => [...prev, { type: 'bot', text: response.data.answer }]);
    } catch (err) {
        setError(err.response ? err.response.data.message : 'Failed to get an answer.');
    } finally {
        setIsLoading(false);
    }
  };

  return (
    <div>
      {!sessionId ? (
        <DocumentUpload onUpload={handleUpload} isLoading={isLoading} />
      ) : (
        // ✅ Replace the old form with your clean ChatBox component
        <ChatBox
          messages={chatHistory}
          input={question}
          setInput={setQuestion}
          handleSend={handleAskQuestion}
          isLoading={isLoading}
        />
      )}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
    </div>
  );
};

export default QaContainer;