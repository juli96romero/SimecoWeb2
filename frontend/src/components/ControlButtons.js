import React from "react";

const ControlButtons = ({ onGenerate, onReset }) => {
  return (
    <div className="button-container">
      <button id="sendMessageButton" onClick={onGenerate}>
        Generar
      </button>
      <button id="resetButton" onClick={onReset}>
        Reset
      </button>
    </div>
  );
};

export default ControlButtons;