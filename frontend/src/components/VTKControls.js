// src/components/VTKControls.js
import React from 'react';

const VTKControls = ({ 
  xValue, 
  setXValue, 
  yValue, 
  setYValue, 
  zValue, 
  setZValue, 
  onButtonPressed 
}) => {
  return (
    <div
      id="controls"
      style={{ paddingTop: "10px", textAlign: "center", position: "absolute", bottom: "10px", width: "100%" }}
    >
      <button id="sendMessageButton" onClick={onButtonPressed}>
        Generar
      </button>
      <div id="inputValues" style={{ marginTop: "10px" }}>
        <label htmlFor="xValue">Valor X:</label>
        <input
          type="number"
          id="xValue"
          value={xValue}
          onChange={(e) => setXValue(parseFloat(e.target.value))}
          style={{ marginRight: "10px", width: "60px" }}
        />
        <label htmlFor="yValue">Valor Y:</label>
        <input
          type="number"
          id="yValue"
          value={yValue}
          onChange={(e) => setYValue(parseFloat(e.target.value))}
          style={{ marginRight: "10px", width: "60px" }}
        />
        <label htmlFor="zValue">Valor Z:</label>
        <input
          type="number"
          id="zValue"
          value={zValue}
          onChange={(e) => setZValue(parseFloat(e.target.value))}
          style={{ width: "60px" }}
        />
      </div>
    </div>
  );
};

export default VTKControls;
