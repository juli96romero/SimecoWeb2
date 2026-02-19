import React, { useEffect, useRef, useCallback } from "react";
import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkFullScreenRenderWindow from "@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkSTLReader from "@kitware/vtk.js/IO/Geometry/STLReader";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import meshColors from "./meshColors";
import { get } from "@kitware/vtk.js/macros";

const VTKViewer = ({ containerRef, onSpecialActorPositionChange, onSpecialActorRotationChange, specialActorPosition, specialActorRotation }) => {
  const context = useRef(null);
  const specialActorRef = useRef(null); // Referencia para el actor especial

  // Función para obtener la posición del actor especial
  const getSpecialActorPosition = useCallback(() => {
    if (specialActorRef.current) {
      const position = specialActorRef.current.getPosition();
      console.log("Posición actual del actor especial:", position); // Debug
      return position;
    }
    return null;
  }, []);

  const getSpecialActorRotation = useCallback(() => {
    if (specialActorRef.current) {
      const rotation = specialActorRef.current.getRotation();
      console.log("Posición actual del actor especial:", rotation); // Debug
      return rotation;
    }
    return null;
  }, []);

  // Efecto para notificar al padre cuando la posición cambia
  useEffect(() => {
    if (onSpecialActorPositionChange) {
      onSpecialActorPositionChange(getSpecialActorPosition());
    }
    if (onSpecialActorRotationChange) {
      onSpecialActorRotationChange(getSpecialActorRotation());
    }
  }, [onSpecialActorPositionChange, getSpecialActorPosition, onSpecialActorRotationChange, getSpecialActorRotation]);

  useEffect(() => {
    if (specialActorRef.current && specialActorPosition) {
      console.log("Nueva posición recibida en useEffect:", specialActorPosition);
      
      // Verificar que la posición es válida
      if (Array.isArray(specialActorPosition)) {
        specialActorRef.current.setPosition(...specialActorPosition);
        specialActorRef.current.setOrientation(...specialActorRotation);
        console.log("Posición del actor especial actualizada:", 
          specialActorRef.current.getPosition());
        console.log("rotacion actual del actor especial:", 
          specialActorRef.current.getOrientation());
        // Forzar el rerenderizado de la escena
        
        if (context.current?.renderWindow) {
          context.current.renderWindow.render();
          console.log("Escena rerenderizada");
        }
      } else {
        console.error("Posición no válida:", specialActorPosition);
      }
    }
  }, [specialActorPosition]);

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

    const initializeVTK = async () => {
      if (!context.current && containerRef.current) {
        // Crear el renderizador de pantalla completa
        const fullScreenRenderer = vtkFullScreenRenderWindow.newInstance({
          rootContainer: containerRef.current,
          containerStyle: { width: "100%", height: "100%" },
        });

        const renderer = fullScreenRenderer.getRenderer();
        const camera = renderer.getActiveCamera();

        // 1. Definir la posición (X, Y, Z)
        camera.setPosition(1, 1, 1); 

        // 2. Definir hacia dónde mira (X, Y, Z)
        camera.setFocalPoint(0, 0, 0); 

        // 3. Definir qué eje es "arriba" (usualmente Z es [0,0,1] o Y es [0,1,0])
        camera.setViewUp(0, 0, 1); 

        // IMPORTANTE: Si usas renderer.resetCamera() después de esto, 
        // vtk anulará tu posición manual para encuadrar todo. 
        // Llama a resetCamera() SOLO si quieres que vtk calcule la distancia automáticamente.
        renderer.resetCamera();
        const renderWindow = fullScreenRenderer.getRenderWindow();

        context.current = {
          fullScreenRenderer,
          renderWindow,
          renderer,
        };

        // Cargar los archivos STL
        const { files } = await fetchStlFiles();
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
              let assignedColor = [1.0, 1.0, 1.0]; // Color por defecto (blanco)
              for (const keyword in meshColors) {
                if (name.includes(keyword.toLowerCase())) {
                  assignedColor = meshColors[keyword];
                  break;
                }
              }

              actor.getProperty().setColor(...assignedColor);
              actor.getProperty().setDiffuseColor(...assignedColor);

              // Verificar si es el archivo especial
              if (name.includes("transductor y fov")) {
                specialActorRef.current = actor; // Almacenar el actor especial
                console.log("Actor especial asignado:", specialActorRef.current); // Debug
                if (onSpecialActorPositionChange) {
                  // Notificar al padre la posición inicial
                  onSpecialActorPositionChange(actor.getPosition());
                }
                if (onSpecialActorRotationChange) {
                  // Notificar al padre la posición inicial
                  onSpecialActorRotationChange(actor.getOrientation());
                }
              }

              // Agregar el actor al renderizador
              renderer.addActor(actor);
            })
            .catch((error) => {
              console.error(`Error loading ${file}:`, error);
            });
        });

        // Renderizar la escena
        renderWindow.render();
      }
    };

    initializeVTK();

    // Limpieza al desmontar el componente
    return () => {
      if (context.current) {
        context.current.fullScreenRenderer.delete();
        context.current = null;
      }
    };
  }, [containerRef, onSpecialActorPositionChange, onSpecialActorRotationChange]);

  return null;
};

export default VTKViewer;