// src/components/ImagePanel.js
import React from 'react';

const ImagePanel = ({ imageData }) => {
  return (
    <div className="half-screen half-screen-left" style={{ flex: 1, display: "flex", flexDirection: "column" }}>
      <div id="imageContainer" style={{ flex: 1, position: "relative" }}>
        <div id="messages"></div>
        <img
          id="receivedImage"
          src={imageData}
          style={{ display: imageData ? "block" : "none", width: "100%", height: "100%", objectFit: "cover" }}
          alt="Received Image"
        />
      </div>
    </div>
  );
};

export default ImagePanel;
