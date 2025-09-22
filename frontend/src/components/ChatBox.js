// components/ChatBox.js
import React from 'react';
import './ChatBox.css';

// ✅ The component now receives all its data and functions as props
const ChatBox = ({
  messages,
  input,
  setInput,
  handleSend,
  isLoading,
}) => {
  const handleFormSubmit = (e) => {
    e.preventDefault(); // Prevent page reload
    handleSend();
  };

  return (
    <div className="chatbox">
      <div className="chatbox-messages">
        {/* It now maps over the 'messages' prop */}
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`chat-message ${msg.type === 'user' ? 'user' : 'bot'}`} // Assuming type is 'user' or 'bot'
          >
            <strong>{msg.type === 'user' ? 'You' : 'Bot'}:</strong> {msg.text}
          </div>
        ))}
      </div>
      {/* The input area is now a form */}
      <form className="chatbox-input" onSubmit={handleFormSubmit}>
        <input
          type="text"
          value={input} // Value comes from props
          onChange={(e) => setInput(e.target.value)} // The function to change the value comes from props
          placeholder="Type your message..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Thinking...' : 'Send'}
        </button>
      </form>
    </div>
  );
};

export default ChatBox;