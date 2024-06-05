import React, { useState, useEffect, useRef } from "react";

const SimecoApp = () => {
  const [xValue, setXValue] = useState(0.58);
  const [yValue, setYValue] = useState(0);
  const [zValue, setZValue] = useState(0.83);

  const xRef = useRef(xValue);
  const yRef = useRef(yValue);
  const zRef = useRef(zValue);

  const url = `ws://${window.location.host}/ws/socket-server/`;
  const chatSocket = new WebSocket(url);
  let buttonState = false;

  useEffect(() => {
    xRef.current = xValue;
    yRef.current = yValue;
    zRef.current = zValue;
  }, [xValue, yValue, zValue]);

  useEffect(() => {
    const receivedImage = document.getElementById("receivedImage");
    const receivedImage_new = document.getElementById("receivedImage_new");
    const button = document.getElementById("sendMessageButton");
    const slider = document.getElementById("scaleSlider");

    chatSocket.onmessage = function (e) {
      let data = JSON.parse(e.data);
      console.log("Data:", data);

      if (data.image_data) {
        receivedImage.src = "data:image/png;base64," + data.image_data;
        receivedImage.style.display = "block";
        if (buttonState) sendMessage();
      } else {
        console.error("Received empty or undefined image_data");
      }
      if (data.new_image_data) {
        receivedImage_new.src = "data:image/png;base64," + data.new_image_data;
        receivedImage_new.style.display = "block";
        
      } else {
        console.error("Received empty or undefined image_data");
      }
    };

    button.addEventListener("click", buttonPressed);
    slider.oninput = function () {
      const scaleValue = this.value;
      receivedImage.style.transform = "scale(" + scaleValue + ")";
    };

    return () => {
      button.removeEventListener("click", buttonPressed);
    };
  }, []);

  const sendMessage = () => {
    chatSocket.send(
      JSON.stringify({
        message: "message",
        x: xRef.current + 0.01 * (Math.random() * 2 - 1),
        y: yRef.current + 0.01 * (Math.random() * 2 - 1),
        z: zRef.current + 0.01 * (Math.random() * 2 - 1),
      })
    );
    console.log("mensaje enviado con: ", xRef.current, yRef.current, zRef.current);
  };

  const buttonPressed = () => {
    buttonState = !buttonState;
    sendMessage();
  };

  return (
    <div>
      <div id="titleAndButton">
        <h1>Simeco Web</h1>
        <button id="sendMessageButton">Generar</button>
        <input
          type="range"
          min="1"
          max="5"
          defaultValue="2"
          step="0.1"
          id="scaleSlider"
        />
      </div>
      <div id="inputValues">
        <label htmlFor="xValue">Valor X:</label>
        <input
          type="number"
          id="xValue"
          value={xValue}
          onChange={(e) => setXValue(parseFloat(e.target.value))}
        />
        <label htmlFor="yValue">Valor Y:</label>
        <input
          type="number"
          id="yValue"
          value={yValue}
          onChange={(e) => setYValue(parseFloat(e.target.value))}
        />
        <label htmlFor="zValue">Valor Z:</label>
        <input
          type="number"
          id="zValue"
          value={zValue}
          onChange={(e) => setZValue(parseFloat(e.target.value))}
        />
      </div>
      <div id="imageContainer">
        <div id="messages"></div>
        <img id="receivedImage" style={{ display: "none" }} alt="Received Image" />
      </div>
      <div id="imageContainer_new">
        <img id="receivedImage_new" style={{ display: "none" }} alt="Received Image New" />
      </div>
    </div>
  );
};

export default SimecoApp;
