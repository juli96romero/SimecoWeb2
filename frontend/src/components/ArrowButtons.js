import React from "react";

const ArrowButtons = ({ onArrowClick }) => {
  return (
    <div className="arrow-buttons">
      <button className="arrow-button" onClick={() => onArrowClick("right", "move")}>
        ↑
      </button>
      <button className="arrow-button" onClick={() => onArrowClick("left", "move")}>
        ↓
      </button>
      <button className="arrow-button" onClick={() => onArrowClick("up", "move")}>
        ←
      </button>
      <button className="arrow-button" onClick={() => onArrowClick("down", "move")}>
        →
      </button>
      
      {/* Nuevos botones para rotación */}
      <button className="arrow-button" onClick={() => onArrowClick("roll_left", "rotate")}>
        1
      </button>
      <button className="arrow-button" onClick={() => onArrowClick("roll_right", "rotate")}>
        3
      </button>
      <button className="arrow-button" onClick={() => onArrowClick("pitch_up", "rotate")}>
        7
      </button>
      <button className="arrow-button" onClick={() => onArrowClick("pitch_down", "rotate")}>
        9
      </button>
      <button className="arrow-button" onClick={() => onArrowClick("yaw_left", "rotate")}>
        4
      </button>
      <button className="arrow-button" onClick={() => onArrowClick("yaw_right", "rotate")}>
        6
      </button>
    </div>
  );
};

export default ArrowButtons;