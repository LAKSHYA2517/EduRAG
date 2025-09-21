import React, { useState } from "react";
import PDFUpload from "../components/PDFUpload";
import ChatBox from "../components/ChatBox";
import "./Home.css";

const Home = () => {
  const [pdfUploaded, setPdfUploaded] = useState(false);

  const handleUpload = () => setPdfUploaded(true);
  const handleBack = () => setPdfUploaded(false);

  return (
    <div className="home-container">
      {!pdfUploaded && <PDFUpload onUpload={handleUpload} />}

      {pdfUploaded && (
        <div className="chat-fullscreen">
          <button className="back-button" onClick={handleBack}>
            ← Back
          </button>
          <ChatBox fullScreen />
        </div>
      )}
    </div>
  );
};

export default Home;
