import React, { useState, useEffect, useRef, useCallback } from "react";
import VTKViewer from "./VTKViewerMovable";
import ImageDisplay from "./ImageDisplay";
import ControlButtons from "./ControlButtons";
import ArrowButtons from "./ArrowButtons";
import BrightnessSlider from "./BrightnessSlider";

const TryVTK = () => {
  const [xValue, setXValue] = useState(0.3);
  const [yValue, setYValue] = useState(0.3);
  const [zValue, setZValue] = useState(0.99);
  const [imageData, setImageData] = useState(null);
  const [imageData2, setImageData2] = useState(null); // Nueva imagen
  const [showImage2, setShowImage2] = useState(false); // Estado para mostrar/ocultar imagen 2
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
  const [specialActorRotation, setSpecialActorRotation] = useState([0, 0, 0]);

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

    setShowImage2(false); // Resetear también el estado de la imagen 2
    setImageData2(null); // Limpiar la imagen 2
    sendMessage("reset", "reset");
  };

  useEffect(() => {
    chatSocket.current = new WebSocket(
      `ws://${window.location.host}/ws/socket-principal-front/`
    );

    chatSocket.current.onmessage = (e) => {
      const data = JSON.parse(e.data);
      console.log('posicion recibida en tryvtk', data.position);
      console.log('rotation', data.rotation);

      if (data.position) {
        // Asegúrate de que data.position es un array de 3 números
        if (Array.isArray(data.position)) {
          setSpecialActorPosition(data.position);
        } else {
          console.error("Posición recibida no es un array válido:", data.position);
        }
      }

      if (data.rotation) {
        // Asegúrate de que data.position es un array de 3 números
        if (Array.isArray(data.position)) {
          setSpecialActorRotation(data.rotation);
        } else {
          console.error("Rotation recibida no es un array válido:", data.position);
        }
      }
      
      if (data.image_data) {
        setImageData(`data:image/png;base64,${data.image_data}`);
        if (buttonState.current) sendMessage();
      } else {
        console.error("Received empty or undefined image_data");
      }

      // Recibir la segunda imagen si está disponible
      if (data.image_data_2) {
        setImageData2(`data:image/png;base64,${data.image_data_2}`);
      }
    };

    return () => {
      if (chatSocket.current) chatSocket.current.close();
    };
  }, []);

  // Función para enviar mensajes
  const sendMessage = useCallback((direction = null, action = null) => {
    if (!chatSocket.current || chatSocket.current.readyState !== WebSocket.OPEN) {
        console.error("WebSocket no está conectado");
        return;
    }
    
    // Mensaje especial para reset
    if (action === "reset") {
        console.log("Enviando mensaje RESET a través del WebSocket");
        chatSocket.current.send(
            JSON.stringify({
                direction: "reset",
                action: "reset",
                message: "reset"
            })
        );
        return;
    }
    
    // Mensaje normal para otros casos
    console.log("Enviando mensaje a través del WebSocket");
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
            arrowUp: arrowUpCount,
            arrowDown: arrowDownCount,
            arrowLeft: arrowLeftCount,
            arrowRight: arrowRightCount,
            specialActorPosition: specialActorPosition,
            direction: direction,
            action: action,
            show_image_2: showImage2,
        })
    );
  }, [xValue, yValue, zValue, arrowUpCount, arrowDownCount, arrowLeftCount, arrowRightCount, specialActorPosition, showImage2]);

  // Función para manejar clics en las flechas
  const handleArrowClick = useCallback((direction, action) => {
    // Enviar la dirección y acción a través del WebSocket
    sendMessage(direction, action);
  }, [sendMessage]);

  // Manejador de eventos de teclado
  const handleKeyDown = useCallback((event) => {
    // Verificar si la tecla presionada es una de las flechas
    switch (event.key) {
      case "ArrowUp":
        event.preventDefault(); // Prevenir comportamiento por defecto
        handleArrowClick("up", "move");
        break;
      case "ArrowDown":
        event.preventDefault();
        handleArrowClick("down", "move");
        break;
      case "ArrowLeft":
        event.preventDefault();
        handleArrowClick("left", "move");
        break;
      case "ArrowRight":
        event.preventDefault();
        handleArrowClick("right", "move");
        break;
      default:
        break;
    }
  }, [handleArrowClick]);

  // Agregar event listener para teclas
  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    
    // Limpiar el event listener al desmontar el componente
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [handleKeyDown]);

  const buttonPressed = () => {
    buttonState.current = !buttonState.current;
    sendMessage();
  };

  // Función para alternar la visualización de la segunda imagen
  const toggleImage2 = () => {
    setShowImage2(prev => !prev);
    // No es necesario enviar mensaje inmediatamente, se enviará en el próximo ciclo
  };

  return (
    <div className="app-container">
      <div className="half-screen half-screen-left">
        <ImageDisplay imageData={imageData} />
        {/* Mostrar la segunda imagen si está activada */}
        {showImage2 && imageData2 && (
          <div className="image2-container">
            <h3>Imagen 2</h3>
            <ImageDisplay imageData={imageData2} />
          </div>
        )}
      </div>
      <div className="half-screen half-screen-right">
        <div ref={vtkContainerRef} className="vtk-container" />
        <VTKViewer
          containerRef={vtkContainerRef}
          onSpecialActorPositionChange={setSpecialActorPosition}
          onSpecialActorRotationChange={setSpecialActorRotation}
          specialActorPosition={specialActorPosition}
          specialActorRotation={specialActorRotation}
        />
        <div id="controls">
          <ControlButtons onGenerate={buttonPressed} onReset={resetValues} />
          <ArrowButtons onArrowClick={handleArrowClick} />
          {/* Botón para activar/desactivar la segunda imagen */}
          <div className="toggle-button-container">
            <button 
              className={`toggle-button ${showImage2 ? 'active' : ''}`}
              onClick={toggleImage2}
            >
              {showImage2 ? 'Ocultar Imagen 2' : 'Mostrar Imagen 2'}
            </button>
          </div>
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