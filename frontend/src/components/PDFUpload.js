import React, { useRef } from "react";
import "./PDFUpload.css";

const PDFUpload = ({ onUpload }) => {
  const fileInputRef = useRef();

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // You can handle file here if needed
      onUpload(); // trigger upload action
    }
  };

  return (
    <div className="upload-wrapper">
      <h3>Upload Your PDF</h3>

      {/* Hidden file input */}
      <input
        type="file"
        accept=".pdf"
        ref={fileInputRef}
        onChange={handleFileChange}
      />

      {/* Single button triggers file picker */}
      <button
        className="file-button"
        onClick={() => fileInputRef.current.click()}
      >
        Choose & Upload
      </button>
    </div>
  );
};

export default PDFUpload;
