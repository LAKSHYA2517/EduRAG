import React, { useRef, useState } from "react";
import "./PDFUpload.css";

const PDFUpload = ({ onUpload }) => {
  const fileInputRef = useRef();
  const [file, setFile] = useState(null);

  const handleButtonClick = () => {
    if (!file) {
      fileInputRef.current.click(); // open file selector
    } else {
      onUpload(file); // upload the selected file
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  return (
    <div className="upload-wrapper">
      <h3>
        {" "}
        Upload your PDFs and get instant <br />
        insights, notes, and answers.
      </h3>
      <input
        type="file"
        accept=".pdf"
        ref={fileInputRef}
        style={{ display: "none" }}
        onChange={handleFileChange}
      />
      <button className="upload-btn" onClick={handleButtonClick}>
        {file ? `Upload ${file.name}` : "Choose File"}
      </button>
    </div>
  );
};

export default PDFUpload;
