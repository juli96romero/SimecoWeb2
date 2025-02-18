import React, { useState, useEffect, useRef } from "react";
import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkFullScreenRenderWindow from "@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkSTLReader from "@kitware/vtk.js/IO/Geometry/STLReader";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import meshColors from "./meshColors";
import "./TryVTK.css";

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

  const brightnessGeneralRef = useRef(brightnessGeneral);
  const brightness1Ref = useRef(brightness1);
  const brightness2Ref = useRef(brightness2);
  const brightness3Ref = useRef(brightness3);
  const brightness4Ref = useRef(brightness4);

  const vtkContainerRef = useRef(null);
  const context = useRef(null);
  const chatSocket = useRef(null);
  const buttonState = useRef(false);
  const fovActorRef = useRef(null); // Ref para almacenar el actor de FOV
  const transducerActorRef = useRef(null); // Ref para almacenar el actor del Transductor

  const [arrowUpCount, setArrowUpCount] = useState(0);
  const [arrowDownCount, setArrowDownCount] = useState(0);
  const [arrowLeftCount, setArrowLeftCount] = useState(0);
  const [arrowRightCount, setArrowRightCount] = useState(0);

  const arrowUpRef = useRef(arrowUpCount);
  const arrowDownRef = useRef(arrowDownCount);
  const arrowLeftRef = useRef(arrowLeftCount);
  const arrowRightRef = useRef(arrowRightCount);

  const handleArrowClick = (direction) => {
    switch (direction) {
      case "up":
        setArrowUpCount(arrowUpCount + 1);
        break;
      case "down":
        setArrowDownCount(arrowDownCount + 1);
        break;
      case "left":
        setArrowLeftCount(arrowLeftCount + 1);
        break;
      case "right":
        setArrowRightCount(arrowRightCount + 1);
        break;
      default:
        break;
    }
    console.log("up",arrowUpCount);
    console.log("down",arrowDownCount);
    console.log("left",arrowLeftCount);
    console.log("rght",arrowRightCount);
    sendArrowCount(direction);
  };

  useEffect(() => {
    brightnessGeneralRef.current = brightnessGeneral;
  }, [brightnessGeneral]);
  
  useEffect(() => {
    brightness1Ref.current = brightness1;
  }, [brightness1]);

  useEffect(() => {
    brightness2Ref.current = brightness2;
  }, [brightness2]);

  useEffect(() => {
    brightness3Ref.current = brightness3;
  }, [brightness3]);

  useEffect(() => {
    brightness4Ref.current = brightness4;
  }, [brightness4]);

  useEffect(() => {
    arrowUpRef.current = arrowUpCount;
  }, [arrowUpCount]);

  useEffect(() => {
    arrowDownRef.current = arrowDownCount;
  }, [arrowDownCount]);

  useEffect(() => {
    arrowLeftRef.current = arrowLeftCount;
  }, [arrowLeftCount]);

  useEffect(() => {
    arrowRightRef.current = arrowRightCount;
  }, [arrowRightCount]);

  useEffect(() => {//websocket init
    chatSocket.current = new WebSocket(
      `ws://${window.location.host}/ws/socket-principal-front/`
    );

    chatSocket.current.onmessage = (e) => {
      const data = JSON.parse(e.data);
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

  const buttonPressed = () => {
    buttonState.current = !buttonState.current;
    sendMessage();
  };

  const sendMessage = () => {
    //sconsole.log('bright', brightnessGeneralRef , brightness1Ref, brightness2Ref, brightness3Ref, brightness4Ref)
    console.log("arrow",arrowUpRef, arrowDownRef,arrowLeftRef,arrowRightRef);
    chatSocket.current.send(
      JSON.stringify({
        message: "message",
        x: xValue + 0.01 * (Math.random() * 2 - 1),
        y: yValue + 0.01 * (Math.random() * 2 - 1),
        z: zValue + 0.01 * (Math.random() * 2 - 1),
        brightness: brightnessGeneralRef.current,
        brightness1: brightness1Ref.current,
        brightness2: brightness2Ref.current,
        brightness3: brightness3Ref.current,
        brightness4: brightness4Ref.current,
        
      })
    );
  };

  useEffect(() => {
    const fetchStlFiles = async () => {
      try {
        const response = await fetch("/api/stl-files/");
        const data = await response.json();
        return data;
      } catch (error) {
        console.error("Error fetching STL files:", error);
        return { files: [], fov: [] };
      }
    };

    if (!context.current && vtkContainerRef.current) {
      const fullScreenRenderer = vtkFullScreenRenderWindow.newInstance({
        rootContainer: vtkContainerRef.current,
        containerStyle: { width: "100%", height: "100%" }, // Ajustar altura al 100%
      });

      const renderer = fullScreenRenderer.getRenderer();
      const renderWindow = fullScreenRenderer.getRenderWindow();

      context.current = {
        fullScreenRenderer,
        renderWindow,
        renderer,
      };

      fetchStlFiles().then(({ files, fov }) => {
        files.forEach((file) => {
          console.log(file);
          const reader = vtkSTLReader.newInstance();
          reader
            .setUrl(`/static/${file}`)
            .then(() => {
              const source = reader.getOutputData(0);
              const mapper = vtkMapper.newInstance();
              mapper.setInputData(source);
              mapper.setScalarVisibility(false);
              const actor = vtkActor.newInstance();
              actor.setMapper(mapper);

              const name = file.toLowerCase();
              let assignedColor = [1.0, 1.0, 1.0];
              for (const keyword in meshColors) {
                if (name.includes(keyword.toLowerCase())) {
                  assignedColor = meshColors[keyword];
                  break;
                }
              }

              actor.getProperty().setColor(...assignedColor);
              actor.getProperty().setDiffuseColor(...assignedColor);
              renderer.addActor(actor);
            })
            .catch((error) => {
              console.error(`Error loading ${file}:`, error);
            });
        });
        renderWindow.render();
      });
    }
  }, []);

  return (
    <div className="app-container">
      <div className="half-screen half-screen-left">
        <div id="imageContainer">
          <div id="messages"></div>
          <img
            id="receivedImage"
            src={imageData}
            className={imageData ? "visible" : "hidden"}
            alt="Received Image"
          />
        </div>
      </div>
      <div className="half-screen half-screen-right">
        <div ref={vtkContainerRef} className="vtk-container" />
        <div id="controls">
          <button id="sendMessageButton" onClick={buttonPressed}>
            Generar
          </button>
          <div className="slider-container">
            <label htmlFor="brightnessGeneral">Brillo General</label>
            <input
              type="range"
              id="brightnessGeneral"
              min="-255"
              max="255"
              step="1"
              value={brightnessGeneral}
              onChange={(e) => setBrightnessGeneral(Number(e.target.value))}
            />
            <label htmlFor="brightness1">Brillo 1</label>
            <input
              type="range"
              id="brightness1"
              min="-255"
              max="255"
              step="1"
              value={brightness1}
              onChange={(e) => setBrightness1(Number(e.target.value))}
            />
            <label htmlFor="brightness2">Brillo 2</label>
            <input
              type="range"
              id="brightness2"
              min="-255"
              max="255"
              step="1"
              value={brightness2}
              onChange={(e) => setBrightness2(Number(e.target.value))}
            />
            <label htmlFor="brightness3">Brillo 3</label>
            <input
              type="range"
              id="brightness3"
              min="-255"
              max="255"
              step="1"
              value={brightness3}
              onChange={(e) => setBrightness3(Number(e.target.value))}
            />
            <label htmlFor="brightness4">Brillo 4</label>
            <input
              type="range"
              id="brightness4"
              min="-255"
              max="255"
              step="1"
              value={brightness4}
              onChange={(e) => setBrightness4(Number(e.target.value))}
            />
          </div>
          <div className="arrow-buttons">
            <button className="arrow-button" onClick={() => handleArrowClick("up")}>
              ↑
            </button>
            <button className="arrow-button" onClick={() => handleArrowClick("down")}>
              ↓
            </button>
            <button className="arrow-button" onClick={() => handleArrowClick("left")}>
              ←
            </button>
            <button className="arrow-button" onClick={() => handleArrowClick("right")}>
              →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TryVTK;
