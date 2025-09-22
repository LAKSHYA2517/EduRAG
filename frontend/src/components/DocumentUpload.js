// components/DocumentUpload.js
import React, { useRef, useState } from "react";
import "./DocumentUpload.css";

const DocumentUpload = ({ onUpload, isLoading }) => {
  const fileInputRef = useRef();
  const [file, setFile] = useState(null);

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      onUpload(selectedFile);
    }
  };

  return (
    <div className="upload-wrapper">
      <h3>
        Upload your document and get instant <br />
        insights, notes, and answers.
      </h3>
      <input
        type="file"
        accept=".ppt, .pptx, .pdf, .doc, .docx"
        ref={fileInputRef}
        style={{ display: "none" }}
        onChange={handleFileChange}
      />
      <button
        className="upload-btn"
        onClick={handleButtonClick}
        disabled={isLoading}
      >
        {isLoading ? "Processing..." : "Choose File to Upload"}
      </button>
    </div>
  );
};

export default DocumentUpload;
