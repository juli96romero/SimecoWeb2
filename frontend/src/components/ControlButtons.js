import React from "react";

const ControlButtons = ({ isRunning, onGenerate, onReset }) => {
  return (
    <div className="button-container">
      <button id="sendMessageButton" onClick={onGenerate}>
        {isRunning ? "Detener" : "Generar"}
      </button>
      <button id="resetButton" onClick={onReset}>
        Reset
      </button>
    </div>
  );
};

export default ControlButtons;