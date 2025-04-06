import React from "react";

const ImageDisplay = ({ imageData }) => {
  return (
    <div id="imageContainer">
      <div id="messages"></div>
      <img
        id="receivedImage"
        src={imageData}
        className={imageData ? "visible" : "hidden"}
        alt="Received Image"
      />
    </div>
  );
};

export default ImageDisplay;