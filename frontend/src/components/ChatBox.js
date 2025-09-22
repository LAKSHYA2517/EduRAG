// components/ChatBox.js
import React from "react";
import "./ChatBox.css";

const ChatBox = ({ messages = [], input, setInput, handleSend, isLoading }) => {
  const handleFormSubmit = (e) => {
    e.preventDefault();
    handleSend();
  };

  return (
    <div className="chatbox">
      <div className="chatbox-messages">
        {(messages || []).map((msg, index) => (
          <div
            key={index}
            className={`chat-message ${msg.type === "user" ? "user" : "bot"}`}
          >
            <strong>{msg.type === "user" ? "You" : "Bot"}:</strong> {msg.text}
          </div>
        ))}
      </div>

      <form className="chatbox-input" onSubmit={handleFormSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? "Thinking..." : "Send"}
        </button>
      </form>
    </div>
  );
};

export default ChatBox;
