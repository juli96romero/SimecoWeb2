// src/components/ButtonPanel.js
import React from 'react';

const ButtonPanel = ({ onMove }) => {
  return (
    <div className="button-panel">
      <button onClick={() => onMove('up')}>Move Up</button>
      <button onClick={() => onMove('down')}>Move Down</button>
      <button onClick={() => onMove('left')}>Move Left</button>
      <button onClick={() => onMove('right')}>Move Right</button>
      <button onClick={() => onMove('rotateClockwise')}>Rotate CW</button>
      <button onClick={() => onMove('rotateCounterClockwise')}>Rotate CCW</button>
    </div>
  );
};

export default ButtonPanel;
