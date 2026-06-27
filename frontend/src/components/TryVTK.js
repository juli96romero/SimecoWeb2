import React, { useState, useEffect, useRef, useCallback } from "react";
import VTKViewerMovable from "./VTKViewerMovable";
import ImageDisplay from "./ImageDisplay";
import ControlButtons from "./ControlButtons";
import ArrowButtons from "./ArrowButtons";
import BrightnessSlider from "./BrightnessSlider";

const TryVTK = () => {
  const [imageData, setImageData] = useState(null);
  const [segmentationImageData, setSegmentationImageData] = useState(null); // VTK slice (split)
  const [overlayImageData, setOverlayImageData] = useState(null); // overlay image
  const [showSecondaryImage, setShowSecondaryImage] = useState(false); // show/hide VTK slice
  const [showOverlay, setShowOverlay] = useState(false); // show/hide overlay

  const [generalGain, setGeneralGain] = useState(0);
  const [bandGain1, setBandGain1] = useState(0);
  const [bandGain2, setBandGain2] = useState(0);
  const [bandGain3, setBandGain3] = useState(0);
  const [bandGain4, setBandGain4] = useState(0);
  const [bandGain5, setBandGain5] = useState(0);
  const [bandGain6, setBandGain6] = useState(0);
  const [bandGain7, setBandGain7] = useState(0);
  const [bandGain8, setBandGain8] = useState(0);

  const [transducerPosition, setTransducerPosition] = useState([0, 0, 0]);
  const [transducerRotation, setTransducerRotation] = useState([0, 0, 0]);

  const brightnessLevelsRef = useRef([
    generalGain,
    bandGain1,
    bandGain2,
    bandGain3,
    bandGain4,
    bandGain5,
    bandGain6,
    bandGain7,
    bandGain8,
  ]);

  const vtkContainerRef = useRef(null);
  const chatSocket = useRef(null);
  const isLoopRunning = useRef(false);
  const [isRunning, setIsRunning] = useState(false);

  const resetValues = () => {
    setGeneralGain(0);
    setBandGain1(0);
    setBandGain2(0);
    setBandGain3(0);
    setBandGain4(0);
    setBandGain5(0);
    setBandGain6(0);
    setBandGain7(0);
    setBandGain8(0);
    brightnessLevelsRef.current = [0, 0, 0, 0, 0, 0, 0, 0, 0];

    setShowSecondaryImage(false);
    setSegmentationImageData(null);
    setShowOverlay(false);
    setOverlayImageData(null);

    sendMessage("reset", "reset");
  };

  useEffect(() => {
    chatSocket.current = new WebSocket(
      `ws://${window.location.host}/ws/socket-principal-front/`
    );

    chatSocket.current.onmessage = (e) => {
      const data = JSON.parse(e.data);
      console.log("position received in tryvtk", data.position);
      console.log("rotation", data.rotation);

      if (data.position) {
        if (Array.isArray(data.position)) {
          setTransducerPosition(data.position);
        } else {
          console.error("Received position is not a valid array:", data.position);
        }
      }

      if (data.rotation) {
        if (Array.isArray(data.rotation)) {
          setTransducerRotation(data.rotation);
        } else {
          console.error("Received rotation is not a valid array:", data.rotation);
        }
      }

      if (data.imageData) {
        setImageData(`data:image/jpeg;base64,${data.imageData}`);
        if (isLoopRunning.current) sendMessage();
      } else {
        console.error("Received empty or undefined imageData");
      }

      if (data.segmentationImageData) {
        setSegmentationImageData(`data:image/jpeg;base64,${data.segmentationImageData}`);
      }

      if (data.overlayImageData) {
        setOverlayImageData(`data:image/jpeg;base64,${data.overlayImageData}`);
      }
    };

    return () => {
      if (chatSocket.current) chatSocket.current.close();
    };
  }, []);

  const sendMessage = useCallback((direction = null, action = null) => {
    if (!chatSocket.current || chatSocket.current.readyState !== WebSocket.OPEN) {
      console.error("WebSocket is not connected");
      return;
    }

    if (action === "reset") {
      console.log("Sending RESET message through the WebSocket");
      chatSocket.current.send(
        JSON.stringify({
          direction: "reset",
          action: "reset",
          message: "reset",
        })
      );
      return;
    }

    console.log("Sending message through the WebSocket");
    chatSocket.current.send(
      JSON.stringify({
        message: "message",
        brightness: brightnessLevelsRef.current[0],
        brightness1: brightnessLevelsRef.current[1],
        brightness2: brightnessLevelsRef.current[2],
        brightness3: brightnessLevelsRef.current[3],
        brightness4: brightnessLevelsRef.current[4],
        brightness5: brightnessLevelsRef.current[5],
        brightness6: brightnessLevelsRef.current[6],
        brightness7: brightnessLevelsRef.current[7],
        brightness8: brightnessLevelsRef.current[8],
        direction: direction,
        action: action,
      })
    );
  }, []);

  const handleArrowClick = useCallback((direction, action) => {
    sendMessage(direction, action);
  }, [sendMessage]);

  const toggleGenerate = () => {
    setIsRunning((prev) => {
      const next = !prev;
      isLoopRunning.current = next;
      sendMessage();
      return next;
    });
  };

  const toggleSecondaryImage = () => {
    setShowSecondaryImage((prev) => !prev);
  };

  const toggleOverlay = () => {
    setShowOverlay((prev) => !prev);
  };

  const handleGainChange = (index, value) => {
    const updatedLevels = [...brightnessLevelsRef.current];
    updatedLevels[index] = value;
    brightnessLevelsRef.current = updatedLevels;
    // resend to refresh the image even if the transducer is idle
    sendMessage();
  };

  const bandGainSetters = [
    setBandGain1, setBandGain2, setBandGain3, setBandGain4,
    setBandGain5, setBandGain6, setBandGain7, setBandGain8,
  ];

  return (
    <div className="app-container">
      <div className={`eco-viewport ${showSecondaryImage ? "split" : "centered"}`}>
        <div className="eco-frame">
          <div style={{ position: "relative", width: "100%", height: "100%" }}>
            <ImageDisplay imageData={imageData} />
            {showOverlay && overlayImageData && (
              <div style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }}>
                <ImageDisplay imageData={overlayImageData} />
              </div>
            )}
          </div>
        </div>

        {showSecondaryImage && segmentationImageData && (
          <div className="eco-secondary">
            <ImageDisplay imageData={segmentationImageData} />
          </div>
        )}
      </div>
      <div className="half-screen half-screen-right">
        <div ref={vtkContainerRef} className="vtk-container" />
        <VTKViewerMovable
          containerRef={vtkContainerRef}
          onTransducerPositionChange={setTransducerPosition}
          onTransducerRotationChange={setTransducerRotation}
          transducerPosition={transducerPosition}
          transducerRotation={transducerRotation}
        />
        <div className="controls-wrapper">
          <div id="controls">
            <section className="controls-section">
              <h4>Simulación</h4>
              <ControlButtons
                isRunning={isRunning}
                onGenerate={toggleGenerate}
                onReset={resetValues}
              />
              <button
                className={`toggle-button ${showSecondaryImage ? "active" : ""}`}
                onClick={toggleSecondaryImage}
              >
                Imagen base
              </button>
              <button
                className={`toggle-button ${showOverlay ? "active" : ""}`}
                onClick={toggleOverlay}
              >
                Superposición
              </button>
            </section>

            <section className="controls-section">
              <h4>Transductor</h4>
              <ArrowButtons onArrowClick={handleArrowClick} />
            </section>

            <section className="controls-section">
              <h4>Ajustes de imagen</h4>
              <BrightnessSlider
                label="Ganancia"
                value={generalGain}
                onChange={(value) => {
                  setGeneralGain(value);
                  handleGainChange(0, value);
                }}
              />
              {[...Array(8)].map((_, i) => (
                <BrightnessSlider
                  key={i}
                  label={`Filtro ${i + 1}`}
                  value={brightnessLevelsRef.current[i + 1]}
                  onChange={(value) => {
                    if (bandGainSetters[i]) bandGainSetters[i](value);
                    handleGainChange(i + 1, value);
                  }}
                />
              ))}
            </section>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TryVTK;
