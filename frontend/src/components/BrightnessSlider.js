import React from "react";

const BrightnessSlider = ({ label, value, onChange }) => {
  return (
    <div className="slider-item">
      <label htmlFor={label}>{label}</label>
      <input
        type="range"
        id={label}
        min="-255"
        max="255"
        step="1"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
};

export default BrightnessSlider;