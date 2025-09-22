// components/DocumentUpload.js
import React, { useRef, useState } from "react";
import "./PDFUpload.css"; // You can rename this CSS file too if you like

const DocumentUpload = ({ onUpload, isLoading }) => { // Added isLoading prop
  const fileInputRef = useRef();
  const [file, setFile] = useState(null);

  const handleButtonClick = () => {
    // We don't check for 'file' here anymore. The parent decides when to upload.
    // This button will now only open the file selector.
    fileInputRef.current.click();
  };

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      // Immediately call the onUpload prop when a file is selected
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
        // ✅ Updated to accept all required file types
        accept=".ppt, .pptx, .pdf, .doc, .docx"
        ref={fileInputRef}
        style={{ display: "none" }}
        onChange={handleFileChange}
      />
      <button className="upload-btn" onClick={handleButtonClick} disabled={isLoading}>
        {/* ✅ Updated button text logic */}
        {isLoading ? "Processing..." : "Choose File to Upload"}
      </button>
    </div>
  );
};

export default DocumentUpload;