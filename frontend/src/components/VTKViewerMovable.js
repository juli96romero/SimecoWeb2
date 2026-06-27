import React, { useEffect, useRef } from "react";
import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkFullScreenRenderWindow from "@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkSTLReader from "@kitware/vtk.js/IO/Geometry/STLReader";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkInteractorStyleTrackballCamera from "@kitware/vtk.js/Interaction/Style/InteractorStyleTrackballCamera";
import meshColors from "./meshColors";

const VTKViewerMovable = ({
  containerRef,
  onTransducerPositionChange,
  onTransducerRotationChange,
  transducerPosition,
  transducerRotation,
}) => {
  const context = useRef(null);
  const transducerRef = useRef(null);
  const skinRef = useRef(null);
  const isInitialized = useRef(false);

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
        console.log("Initializing VTK...");

        const fullScreenRenderer = vtkFullScreenRenderWindow.newInstance({
          rootContainer: containerRef.current,
          containerStyle: { width: "100%", height: "100%" },
        });

        const renderer = fullScreenRenderer.getRenderer();
        const renderWindow = fullScreenRenderer.getRenderWindow();
        const camera = renderer.getActiveCamera();

        // initial camera position
        camera.setPosition(-0.21, -0.22, -4.1);
        camera.setFocalPoint(0, 0, 0);
        camera.setViewUp(0, -1, 0); // the -1 on Y means the scene is flipped relative to the vtk standard

        const interactor = renderWindow.getInteractor();
        context.current = {
          fullScreenRenderer, 
          renderWindow, 
          renderer,
          interactor 
        };

        const files = await fetchStlFiles();
        console.log("STL files found:", files);

        if (files.length === 0) {
          console.warn("No STL files found");
          return;
        }

        files.forEach((file) => {
          const reader = vtkSTLReader.newInstance();
          reader
            .setUrl(`/static/${file}`)
            .then(() => {
              console.log("Name received:", file);
              const source = reader.getOutputData(0);
              const mapper = vtkMapper.newInstance();
              mapper.setInputData(source);
              mapper.setScalarVisibility(false);

              const actor = vtkActor.newInstance();
              actor.setMapper(mapper);

              // assign colors based on the file name
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

              // identify key actors
              if (name.includes("skin")) {
                skinRef.current = actor;
                actor.getProperty().setOpacity(0.7); // make skin semi-transparent
                console.log("Skin identified:", file);
              }
              if (name.includes("transductor")) {
                transducerRef.current = actor;
                // highlight the transducer
                actor.getProperty().setColor(1, 0, 0); // red
                actor.setPosition(0, 0, -0.85);
                console.log("rotation:", actor.getOrientation());
                actor.setOrientation(0.0, 180, 0.0);
                console.log("Transducer identified:", file);

                // notify the initial position
                if (onTransducerPositionChange) {
                  onTransducerPositionChange(actor.getPosition());
                }
                if (onTransducerRotationChange) {
                  onTransducerRotationChange(actor.getOrientation());
                }
              }

              renderer.addActor(actor);
              console.log("Actor added:", file);

              // render after adding each actor
              renderWindow.render();
            })
            .catch((error) => {
              console.error(`Error loading ${file}:`, error);
            });
        });

        // configure interaction after loading the models
        const style = vtkInteractorStyleTrackballCamera.newInstance();
        interactor.setInteractorStyle(style);

        // configure interaction for transducer movement
        let dragging = false;
        let lastPos = null;

        interactor.onRightButtonPress((callData) => {
            if (!transducerRef.current) return;
            
            dragging = true;
            lastPos = { x: callData.position.x, y: callData.position.y };
            interactor.requestAnimation(style);
        });

        interactor.onRightButtonRelease(() => {
            dragging = false;
            interactor.cancelAnimation(style);
        });
        interactor.onMouseMove((callData) => {
            if (!dragging || !transducerRef.current || !skinRef.current || !lastPos) return;

            const dx = callData.position.x - lastPos.x;
            const dy = callData.position.y - lastPos.y;
            lastPos = { x: callData.position.x, y: callData.position.y };

            const rotationSpeed = 0.01;
            const skinCenter = [0.00664, -0.00142, -0.04225];
            const sphereRadius = 0.75;

            const currentPos = transducerRef.current.getPosition();
            
            // current approximate angles
            const currentAngleX = Math.atan2(currentPos[1] - skinCenter[1], currentPos[0] - skinCenter[0]);
            const currentAngleY = Math.atan2(currentPos[2] - skinCenter[2], 
                                          Math.sqrt((currentPos[0]-skinCenter[0])**2 + (currentPos[1]-skinCenter[1])**2));
            
            // new angles
            const newAngleX = currentAngleX - dx * rotationSpeed;
            const newAngleY = Math.max(-Math.PI/2 + 0.1, Math.min(Math.PI/2 - 0.1, currentAngleY - dy * rotationSpeed));
            
            // new position
            const finalPos = [
                skinCenter[0] + sphereRadius * Math.cos(newAngleY) * Math.cos(newAngleX),
                skinCenter[1] + sphereRadius * Math.cos(newAngleY) * Math.sin(newAngleX),
                skinCenter[2] + sphereRadius * Math.sin(newAngleY)
            ];

            transducerRef.current.setPosition(...finalPos);
            
            if (onTransducerPositionChange) {
                onTransducerPositionChange(finalPos);
            }
            
            renderWindow.render();
        });
        isInitialized.current = true;
        console.log("VTK initialized successfully");

        // final render
        setTimeout(() => {
          renderWindow.render();
        }, 100);

      } catch (error) {
        console.error("Error initializing VTK:", error);
      }
    };

    initializeVTK();

    return () => {
      if (context.current) {
        console.log("Cleaning up VTK...");
        context.current.fullScreenRenderer.delete();
        context.current = null;
        isInitialized.current = false;
        transducerRef.current = null;
        skinRef.current = null;
      }
    };
  }, [containerRef, onTransducerPositionChange, onTransducerRotationChange]);

  // sync with the external position
  useEffect(() => {
    if (transducerRef.current && Array.isArray(transducerPosition)) {
      console.log("Updating position from props:", transducerPosition);
      transducerRef.current.setPosition(...transducerPosition);

      if (transducerRotation && Array.isArray(transducerRotation)) {
        transducerRef.current.setOrientation(...transducerRotation);
      }

      if (context.current?.renderWindow) {
        context.current.renderWindow.render();
      }
    }
  }, [transducerPosition, transducerRotation]);

  return null;
};

export default VTKViewerMovable;
