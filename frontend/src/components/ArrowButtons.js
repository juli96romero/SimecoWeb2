import React from "react";

const ArrowButtons = ({ onArrowClick }) => {
  return (
    <div className="transducer-controls">

      {/* MOVIMIENTO */}
      <div className="control-block">
        <div className="control-title">Mover</div>

        <div className="arrow-pad">
          <button onClick={() => onArrowClick("up", "move")}>↑</button>

          <div className="arrow-row">
            <button onClick={() => onArrowClick("left", "move")}>←</button>
            <button onClick={() => onArrowClick("down", "move")}>↓</button>
            <button onClick={() => onArrowClick("right", "move")}>→</button>
          </div>
        </div>
      </div>

      {/* ROTACIÓN */}
      <div className="control-block">
        <div className="control-title">Rotar</div>

        <div className="rotation-controls">
          <div className="rotation-row">
            <span>X</span>
            <button onClick={() => onArrowClick("roll_left", "rotate")}>−</button>
            <button onClick={() => onArrowClick("roll_right", "rotate")}>+</button>
          </div>

          <div className="rotation-row">
            <span>Y</span>
            <button onClick={() => onArrowClick("pitch_down", "rotate")}>−</button>
            <button onClick={() => onArrowClick("pitch_up", "rotate")}>+</button>
          </div>

          <div className="rotation-row">
            <span>Z</span>
            <button onClick={() => onArrowClick("yaw_left", "rotate")}>−</button>
            <button onClick={() => onArrowClick("yaw_right", "rotate")}>+</button>
          </div>
        </div>
      </div>

    </div>
  );
};

export default ArrowButtons;
