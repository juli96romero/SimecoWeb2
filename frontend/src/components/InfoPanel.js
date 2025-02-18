// src/components/InfoPanel.js
import React from 'react';

const InfoPanel = ({ position, angle }) => {
  return (
    <div className="info-panel">
      <h3>Transducer Information</h3>
      <p><strong>Position:</strong> {position.x}, {position.y}, {position.z}</p>
      <p><strong>Angle:</strong> {angle}°</p>
    </div>
  );
};

export default InfoPanel;
