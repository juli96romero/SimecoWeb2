import React, { useState, useEffect, useRef } from "react";
import VTKViewer from "./VTKViewer";
import ImageDisplay from "./ImageDisplay";
import ControlButtons from "./ControlButtons";
import ArrowButtons from "./ArrowButtons";
import BrightnessSlider from "./BrightnessSlider";

const TryVTK = () => {
  const [xValue, setXValue] = useState(0.3);
  const [yValue, setYValue] = useState(0.3);
  const [zValue, setZValue] = useState(0.99);
  const [imageData, setImageData] = useState(null);
  const [brightnessGeneral, setBrightnessGeneral] = useState(0);
  const [brightness1, setBrightness1] = useState(0);
  const [brightness2, setBrightness2] = useState(0);
  const [brightness3, setBrightness3] = useState(0);
  const [brightness4, setBrightness4] = useState(0);
  const [brightness5, setBrightness5] = useState(0);
  const [brightness6, setBrightness6] = useState(0);
  const [brightness7, setBrightness7] = useState(0);
  const [brightness8, setBrightness8] = useState(0);

  // Estado para almacenar la posición del actor especial
  const [specialActorPosition, setSpecialActorPosition] = useState([0, 0, 0]);

  const brightnessRefs = useRef([
    brightnessGeneral,
    brightness1,
    brightness2,
    brightness3,
    brightness4,
    brightness5,
    brightness6,
    brightness7,
    brightness8,
  ]);

  const vtkContainerRef = useRef(null);
  const chatSocket = useRef(null);
  const buttonState = useRef(false);

  // Estado para los contadores de flechas
  const [arrowUpCount, setArrowUpCount] = useState(0);
  const [arrowDownCount, setArrowDownCount] = useState(0);
  const [arrowLeftCount, setArrowLeftCount] = useState(0);
  const [arrowRightCount, setArrowRightCount] = useState(0);

  // Función para manejar clics en las flechas
  const handleArrowClick = (direction) => {
    switch (direction) {
      case "up":
        setArrowUpCount((prev) => prev + 1);
        break;
      case "down":
        setArrowDownCount((prev) => prev + 1);
        break;
      case "left":
        setArrowLeftCount((prev) => prev + 1);
        break;
      case "right":
        setArrowRightCount((prev) => prev + 1);
        break;
      default:
        break;
    }

    // Enviar la dirección de movimiento a través del WebSocket
    sendMessage(direction);
  };

  const resetValues = () => {
    setBrightnessGeneral(0);
    setBrightness1(0);
    setBrightness2(0);
    setBrightness3(0);
    setBrightness4(0);
    setBrightness5(0);
    setBrightness6(0);
    setBrightness7(0);
    setBrightness8(0);
    setArrowUpCount(0);
    setArrowDownCount(0);
    setArrowLeftCount(0);
    setArrowRightCount(0);
  };

  useEffect(() => {
    chatSocket.current = new WebSocket(
      `ws://${window.location.host}/ws/socket-principal-front/`
    );

    chatSocket.current.onmessage = (e) => {
      const data = JSON.parse(e.data);
      console.log(data.position);
      if (data.image_data) {
        setImageData(`data:image/png;base64,${data.image_data}`);
        if (buttonState.current) sendMessage();
      } else {
        console.error("Received empty or undefined image_data");
      }
    };

    return () => {
      if (chatSocket.current) chatSocket.current.close();
    };
  }, []);

  const sendMessage = (direction = null) => {
    chatSocket.current.send(
      JSON.stringify({
        message: "message",
        x: xValue + 0.01 * (Math.random() * 2 - 1),
        y: yValue + 0.01 * (Math.random() * 2 - 1),
        z: zValue + 0.01 * (Math.random() * 2 - 1),
        brightness: brightnessRefs.current[0],
        brightness1: brightnessRefs.current[1],
        brightness2: brightnessRefs.current[2],
        brightness3: brightnessRefs.current[3],
        brightness4: brightnessRefs.current[4],
        brightness5: brightnessRefs.current[5],
        brightness6: brightnessRefs.current[6],
        brightness7: brightnessRefs.current[7],
        brightness8: brightnessRefs.current[8],
        arrowUp: arrowUpCount, // Usar el estado directamente
        arrowDown: arrowDownCount, // Usar el estado directamente
        arrowLeft: arrowLeftCount, // Usar el estado directamente
        arrowRight: arrowRightCount, // Usar el estado directamente
        specialActorPosition: specialActorPosition, // Enviar la posición del actor especial
        direction: direction, // Enviar la dirección de movimiento
      })
    );
  };

  const buttonPressed = () => {
    buttonState.current = !buttonState.current;
    sendMessage();
  };

  return (
    <div className="app-container">
      <div className="half-screen half-screen-left">
        <ImageDisplay imageData={imageData} />
      </div>
      <div className="half-screen half-screen-right">
        <div ref={vtkContainerRef} className="vtk-container" />
        <VTKViewer
          containerRef={vtkContainerRef}
          onSpecialActorPositionChange={setSpecialActorPosition} // Pasar la función
          specialActorPosition={specialActorPosition} // Pasar la posición actualizada
        />
        <div id="controls">
          <ControlButtons onGenerate={buttonPressed} onReset={resetValues} />
          <ArrowButtons onArrowClick={handleArrowClick} />
          <div className="slider-container">
            <BrightnessSlider
              label="Brillo General"
              value={brightnessGeneral}
              onChange={setBrightnessGeneral}
            />
            {[...Array(8)].map((_, i) => (
              <BrightnessSlider
                key={i}
                label={`Brillo ${i + 1}`}
                value={brightnessRefs.current[i + 1]}
                onChange={(value) => {
                  const newBrightness = [...brightnessRefs.current];
                  newBrightness[i + 1] = value;
                  brightnessRefs.current = newBrightness;
                }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TryVTK;