import React, { useState, useEffect, useRef } from "react";
import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkFullScreenRenderWindow from "@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkCellArray from "@kitware/vtk.js/Common/Core/CellArray";
import vtkPoints from "@kitware/vtk.js/Common/Core/Points";
import vtkSTLReader from "@kitware/vtk.js/IO/Geometry/STLReader";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";

const meshColors = {
  pelvis: [151 / 255.0, 151 / 255.0, 147 / 255.0],
  spleen: [1.0, 0, 1.0],
  liver: [100 / 255.0, 0, 100 / 255.0],
  surrenalGland: [0, 1.0, 1.0],
  kidney: [1.0, 1.0, 0],
  gallbladder: [0, 1.0, 0],
  pancreas: [0, 0, 1.0],
  artery: [1.0, 0, 0],
  bones: [1.0, 1.0, 1.0],
};

const TryVTK = () => {
  const [xValue, setXValue] = useState(0.3);
  const [yValue, setYValue] = useState(0.3);
  const [zValue, setZValue] = useState(0.99);
  const [imageData, setImageData] = useState(null);

  const vtkContainerRef = useRef(null);
  const context = useRef(null);
  const chatSocket = useRef(null);
  const buttonState = useRef(false);
  const fovActorRef = useRef(null); // Ref para almacenar el actor de FOV
  const transducerActorRef = useRef(null); // Ref para almacenar el actor del Transductor

  // Estado para la posición y orientación
  const position = useRef([0, 0, 0]);
  const orientation = useRef([0, 0, 0]); // [pitch, yaw, roll]

  // Función para calcular la orientación hacia el centro
  const calculateOrientation = (position) => {
    const [x, y, z] = position;

    // Calcular la distancia en el plano XZ
    const distanceXZ = Math.sqrt(x * x + z * z);

    // Calcular yaw (rotación alrededor del eje Y)
    const yaw = Math.atan2(x, z) * (180 / Math.PI);

    // Calcular pitch (rotación alrededor del eje X)
    const pitch = Math.atan2(y, distanceXZ) * (180 / Math.PI);

    return [pitch, yaw, 0]; // Roll se mantiene en 0
  };

  useEffect(() => {
    chatSocket.current = new WebSocket(`ws://${window.location.host}/ws/socket-principal-front/`);

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

  const sendMessage = () => {
    chatSocket.current.send(
      JSON.stringify({
        message: "message",
        x: xValue + 0.01 * (Math.random() * 2 - 1),
        y: yValue + 0.01 * (Math.random() * 2 - 1),
        z: zValue + 0.01 * (Math.random() * 2 - 1),
      })
    );
  };

  const buttonPressed = () => {
    buttonState.current = !buttonState.current;
    sendMessage();
  };

  const updateActorsPositionAndOrientation = (x, y, z, pitch, yaw, roll) => {
    if (fovActorRef.current) {
      fovActorRef.current.setPosition(x, y, z);
      fovActorRef.current.setOrientation(pitch, yaw, roll);
    }

    if (transducerActorRef.current) {
      // Ajusta la posición relativa según sea necesario
      // Por ejemplo, 1 unidad delante del FOV en el eje Z
      transducerActorRef.current.setPosition(x, y, z + 1);
      transducerActorRef.current.setOrientation(pitch, yaw, roll);
    }

    context.current.renderWindow.render();
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

    const fetchTransducer = async () => {
      try {
        const response = await fetch("/api/stl-transductor/");
        const data = await response.json();
        return data; // Suponiendo que la API devuelve una lista de archivos o una única ruta
      } catch (error) {
        console.error("Error fetching transductor STL file:", error);
        return null;
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

      // Cargar las mallas principales y el transductor en paralelo
      Promise.all([fetchStlFiles(), fetchTransducer()]).then(([stlData, transducerData]) => {
        const { files, fov } = stlData;

        // Cargar mallas principales
        files.forEach((file) => {
          console.log(`Cargando malla: ${file}`);
          const reader = vtkSTLReader.newInstance();
          reader.setUrl(`/static/${file}`).then(() => {
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
          }).catch((error) => {
            console.error(`Error loading ${file}:`, error);
          });
        });

        // Crear la malla de FOV
        const fovSource = vtkPolyData.newInstance();
        const points = vtkPoints.newInstance();
        const cells = vtkCellArray.newInstance();

        fov.forEach((point) => {
          points.insertNextPoint(point.x, point.y, point.z);
        });

        const numPoints = points.getNumberOfPoints();
        const cell = new Uint32Array(numPoints);
        for (let i = 0; i < numPoints; i++) {
          cell[i] = i;
        }

        cells.insertNextCell(cell);

        fovSource.setPoints(points);
        fovSource.setPolys(cells);

        const fovMapper = vtkMapper.newInstance();
        fovMapper.setInputData(fovSource);
        const fovActor = vtkActor.newInstance();
        fovActor.setMapper(fovMapper);
        fovActor.getProperty().setColor(1.0, 0.0, 0.0); // Color rojo

        // Posicionar la malla de FOV inicialmente en el origen
        fovActor.setPosition(0, 0, 0);
        fovActor.setOrientation(0, 0, 0);

        fovActorRef.current = fovActor; // Almacenar el actor de FOV en la ref
        renderer.addActor(fovActor);

        // Cargar el transductor si existe
        if (transducerData && transducerData.file) {
          const transducerFile = transducerData.file; // Suponiendo que la API devuelve { file: 'transductor.stl' }
          console.log(`Cargando transductor: ${transducerFile}`);
          const transducerReader = vtkSTLReader.newInstance();
          transducerReader.setUrl(`/static/${transducerFile}`).then(() => {
            const transducerSource = transducerReader.getOutputData(0);
            const transducerMapper = vtkMapper.newInstance();
            transducerMapper.setInputData(transducerSource);
            transducerMapper.setScalarVisibility(false);
            const transducerActor = vtkActor.newInstance();
            transducerActor.setMapper(transducerMapper);

            transducerActor.getProperty().setColor(0.0, 0.0, 1.0); // Azul

            // Posicionar el transductor relativo al FOV
            transducerActor.setPosition(0, 0, 1); // Ajusta según sea necesario
            transducerActor.setOrientation(0, 0, 0);

            renderer.addActor(transducerActor);
            transducerActorRef.current = transducerActor;

            // Renderizar la escena nuevamente después de agregar el transductor
            renderWindow.render();
          }).catch((error) => {
            console.error(`Error loading transductor ${transducerData.file}:`, error);
          });
        }

        // Ajustar la cámara para un espacio más amplio
        renderer.resetCamera();

        // Obtener la cámara activa
        const camera = renderer.getActiveCamera();

        // Obtener los límites de la escena después de resetCamera
        const bounds = renderer.computeVisiblePropBounds();

        // Ajustar el rango de recorte basado en los límites de la escena
        const padding = 2; // Factor de multiplicación para ampliar el rango
        const maxDistance = Math.max(
          Math.abs(bounds[0]),
          Math.abs(bounds[1]),
          Math.abs(bounds[2]),
          Math.abs(bounds[3]),
          Math.abs(bounds[4]),
          Math.abs(bounds[5])
        ) * padding;

        camera.setClippingRange(0.1, maxDistance);

        // Opcional: Posicionar la cámara más alejada en el eje Z
        camera.setPosition(0, 0, maxDistance * 1.5);
        camera.setFocalPoint(0, 0, 0);
        camera.setViewUp(0, 1, 0);

        // Actualizar el rango de recorte de la cámara
        renderer.resetCameraClippingRange();

        renderWindow.render();
      });

      // Función para manejar teclas de movimiento y rotación
      const handleKeyDown = (event) => {
        const key = event.key.toLowerCase();

        if (fovActorRef.current && transducerActorRef.current) {
          const moveStep = 0.1; // Tamaño del paso para movimiento
          const rotateStep = 5; // Grados de rotación por paso

          let [currentPitch, currentYaw, currentRoll] = orientation.current;
          let [currentX, currentY, currentZ] = position.current;

          switch (key) {
            // Controles de movimiento con WASD QE
            case 'w':
              currentY += moveStep;
              break;
            case 's':
              currentY -= moveStep;
              break;
            case 'a':
              currentX -= moveStep;
              break;
            case 'd':
              currentX += moveStep;
              break;
            case 'q':
              currentZ += moveStep;
              break;
            case 'e':
              currentZ -= moveStep;
              break;

            // Controles de rotación con las flechas
            case 'arrowup':
              currentPitch += rotateStep;
              break;
            case 'arrowdown':
              currentPitch -= rotateStep;
              break;
            case 'arrowleft':
              currentYaw -= rotateStep;
              break;
            case 'arrowright':
              currentYaw += rotateStep;
              break;

            default:
              return; // Salir si la tecla no es relevante
          }

          // Actualizar el estado de posición y orientación
          position.current = [currentX, currentY, currentZ];
          orientation.current = [currentPitch, currentYaw, currentRoll];

          // Aplicar los cambios a los actores
          updateActorsPositionAndOrientation(currentX, currentY, currentZ, currentPitch, currentYaw, currentRoll);
        }
      };

      window.addEventListener("keydown", handleKeyDown);

      // Limpiar el event listener al desmontar el componente
      return () => {
        window.removeEventListener("keydown", handleKeyDown);
        if (context.current) {
          const { fullScreenRenderer, renderer } = context.current;
          renderer.getActors().forEach((actor) => actor.delete());
          fullScreenRenderer.delete();
          context.current = null;
        }
      };
    }}, []);

    return (
      <div className="app-container" style={{ display: "flex", height: "100vh" }}>
        <div className="half-screen half-screen-left" style={{ flex: 3, display: "flex", flexDirection: "column" }}>
          <div id="imageContainer" style={{ flex: 1, position: "relative" }}>
            <div id="messages"></div>
            <img
              id="receivedImage"
              src={imageData}
              style={{ display: imageData ? "block" : "none", width: "100%", height: "100%", objectFit: "cover" }}
              alt="Received Image"
            />
          </div>
        </div>
        <div className="half-screen half-screen-right" style={{ flex: 1, position: "relative" }}>
          <div ref={vtkContainerRef} style={{ width: "100%", height: "100%" }} />
          <div
            id="controls"
            style={{
              paddingTop: "10px",
              textAlign: "center",
              position: "absolute",
              bottom: "10px",
              width: "100%",
            }}
          >
            <button id="sendMessageButton" onClick={buttonPressed}>
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
        </div>
      </div>
    );
  };

  export default TryVTK;
