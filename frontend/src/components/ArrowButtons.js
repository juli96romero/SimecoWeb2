import React from "react";

const ArrowButtons = ({ onArrowClick }) => {
  return (
    <div className="arrow-buttons">
      <button className="arrow-button" onClick={() => onArrowClick("left")}>
        ↑
      </button>
      <button className="arrow-button" onClick={() => onArrowClick("right")}>
        ↓
      </button>
      <button className="arrow-button" onClick={() => onArrowClick("up")}>
        ←
      </button>
      <button className="arrow-button" onClick={() => onArrowClick("down")}>
        →
      </button>
    </div>
  );
};

export default ArrowButtons;