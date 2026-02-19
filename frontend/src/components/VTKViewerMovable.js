import React, { useEffect, useRef, useCallback } from "react";
import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkFullScreenRenderWindow from "@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkSTLReader from "@kitware/vtk.js/IO/Geometry/STLReader";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkInteractorStyleTrackballCamera from "@kitware/vtk.js/Interaction/Style/InteractorStyleTrackballCamera";
import meshColors from "./meshColors";

const VTKViewerMovable = ({
  containerRef,
  onSpecialActorPositionChange,
  onSpecialActorRotationChange,
  specialActorPosition,
  specialActorRotation,
}) => {
  const context = useRef(null);
  const transductorRef = useRef(null);
  const skinRef = useRef(null);
  const isInitialized = useRef(false);

  // Función para obtener la posición del actor especial
  const getSpecialActorPosition = useCallback(() => {
    if (transductorRef.current) {
      return transductorRef.current.getPosition();
    }
    return null;
  }, []);

  const getSpecialActorRotation = useCallback(() => {
    if (transductorRef.current) {
      return transductorRef.current.getOrientation();
    }
    return null;
  }, []);

  useEffect(() => {
    const fetchStlFiles = async () => {
      try {
        const response = await fetch("/api/stl-files/");
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data.files || [];
      } catch (error) {
        console.error("Error fetching STL files:", error);
        return [];
      }
    };

    const initializeVTK = async () => {
      if (isInitialized.current || !containerRef?.current) {
        return;
      }

      try {
        console.log("Inicializando VTK...");
        
        // Usar el mismo enfoque simple que funciona en el segundo código
        const fullScreenRenderer = vtkFullScreenRenderWindow.newInstance({
          rootContainer: containerRef.current,
          containerStyle: { width: "100%", height: "100%" },
        });

        const renderer = fullScreenRenderer.getRenderer();
        const renderWindow = fullScreenRenderer.getRenderWindow();
        const camera = renderer.getActiveCamera(); // <--- Obtenemos la cámara

        // --- CONFIGURACIÓN DE POSICIÓN INICIAL ---
        // Usando tus valores de la prueba 2 (redondeados para limpieza)
        camera.setPosition(-0.21, -0.22, -4.1);
        camera.setFocalPoint(0, 0, 0);
        camera.setViewUp(0, -1, 0); // El -1 en Y indica que tu escena está "invertida" respecto al estándar vtk
        
        const interactor = renderWindow.getInteractor();
        window.myCamera = renderer.getActiveCamera();
        context.current = { 
          fullScreenRenderer, 
          renderWindow, 
          renderer,
          interactor 
        };

        const files = await fetchStlFiles();
        console.log("Archivos STL encontrados:", files);

        if (files.length === 0) {
          console.warn("No se encontraron archivos STL");
          return;
        }

        // Cargar archivos de manera similar al segundo código
        files.forEach((file) => {
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

              // Asignar colores según el nombre del archivo
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

              // Identificar actores clave
              if (name.includes("skin")) {
                skinRef.current = actor;
                actor.getProperty().setOpacity(0.7); // Hacer skin semitransparente
                console.log("Skin identificado:", file);
              }
              if (name.includes("transductor")) {
                transductorRef.current = actor;
                // Destacar el transductor
                actor.getProperty().setColor(1, 0, 0); // Rojo
                actor.setPosition(0, 0, -0.85);
                console.log("rotation:", actor.getOrientation());
                actor.setOrientation(0.0, 180, 0.0);
                console.log("Transductor identificado:", file);
                
                // Notificar posición inicial
                if (onSpecialActorPositionChange) {
                  onSpecialActorPositionChange(actor.getPosition());
                }
                if (onSpecialActorRotationChange) {
                  onSpecialActorRotationChange(actor.getOrientation());
                }
              }

              renderer.addActor(actor);
              console.log("Actor añadido:", file);

              // Renderizar después de agregar cada actor
              renderWindow.render();
            })
            .catch((error) => {
              console.error(`Error loading ${file}:`, error);
            });
        });

        // Configurar interacción DESPUÉS de cargar los modelos
        const style = vtkInteractorStyleTrackballCamera.newInstance();
        interactor.setInteractorStyle(style);

        // Reemplaza la sección de interacción de mouse con este código:

        // Configurar interacción para movimiento del transductor
        let dragging = false;
        let lastPos = null;

        interactor.onRightButtonPress((callData) => {
            if (!transductorRef.current) return;
            
            dragging = true;
            lastPos = { x: callData.position.x, y: callData.position.y };
            interactor.requestAnimation(style);
        });

        interactor.onRightButtonRelease(() => {
            dragging = false;
            interactor.cancelAnimation(style);
        });
        interactor.onMouseMove((callData) => {
            if (!dragging || !transductorRef.current || !skinRef.current || !lastPos) return;

            const dx = callData.position.x - lastPos.x;
            const dy = callData.position.y - lastPos.y;
            lastPos = { x: callData.position.x, y: callData.position.y };

            const rotationSpeed = 0.01;
            const skinCenter = [0.00664, -0.00142, -0.04225];
            const sphereRadius = 0.75;

            const currentPos = transductorRef.current.getPosition();
            
            // Ángulos actuales aproximados
            const currentAngleX = Math.atan2(currentPos[1] - skinCenter[1], currentPos[0] - skinCenter[0]);
            const currentAngleY = Math.atan2(currentPos[2] - skinCenter[2], 
                                          Math.sqrt((currentPos[0]-skinCenter[0])**2 + (currentPos[1]-skinCenter[1])**2));
            
            // Nuevos ángulos
            const newAngleX = currentAngleX - dx * rotationSpeed;
            const newAngleY = Math.max(-Math.PI/2 + 0.1, Math.min(Math.PI/2 - 0.1, currentAngleY - dy * rotationSpeed));
            
            // Nueva posición
            const finalPos = [
                skinCenter[0] + sphereRadius * Math.cos(newAngleY) * Math.cos(newAngleX),
                skinCenter[1] + sphereRadius * Math.cos(newAngleY) * Math.sin(newAngleX),
                skinCenter[2] + sphereRadius * Math.sin(newAngleY)
            ];

            transductorRef.current.setPosition(...finalPos);
            
            if (onSpecialActorPositionChange) {
                onSpecialActorPositionChange(finalPos);
            }
            
            renderWindow.render();
        });
        isInitialized.current = true;
        console.log("VTK inicializado correctamente");

        // Render final
        setTimeout(() => {
          renderWindow.render();
        }, 100);

      } catch (error) {
        console.error("Error inicializando VTK:", error);
      }
    };

    initializeVTK();

    return () => {
      if (context.current) {
        console.log("Limpiando VTK...");
        context.current.fullScreenRenderer.delete();
        context.current = null;
        isInitialized.current = false;
        transductorRef.current = null;
        skinRef.current = null;
      }
    };
  }, [containerRef, onSpecialActorPositionChange, onSpecialActorRotationChange]);

  // Sincronizar con posición externa
  useEffect(() => {
    if (transductorRef.current && Array.isArray(specialActorPosition)) {
      console.log("Actualizando posición desde props:", specialActorPosition);
      transductorRef.current.setPosition(...specialActorPosition);
      
      if (specialActorRotation && Array.isArray(specialActorRotation)) {
        transductorRef.current.setOrientation(...specialActorRotation);
      }
      
      if (context.current?.renderWindow) {
        context.current.renderWindow.render();
      }
    }
  }, [specialActorPosition, specialActorRotation]);

  return null;
};

export default VTKViewerMovable;