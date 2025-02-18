// src/components/VtkComponent.js
import React, { useEffect, useRef } from "react";
import vtkFullScreenRenderWindow from "@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkCellArray from "@kitware/vtk.js/Common/Core/CellArray";
import vtkPoints from "@kitware/vtk.js/Common/Core/Points";
import vtkSTLReader from "@kitware/vtk.js/IO/Geometry/STLReader";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import { meshColors } from "../utils/constants"; // Import colors from a separate file
import { initializeWebSocket } from "../utils/websocket"; // Import WebSocket logic

const VtkComponent = () => {
  const vtkContainerRef = useRef(null);
  const context = useRef(null);
  const fovActorRef = useRef(null);
  
  // Other state variables can be added here as needed

  useEffect(() => {
    const chatSocket = initializeWebSocket();

    if (!context.current && vtkContainerRef.current) {
      const fullScreenRenderer = vtkFullScreenRenderWindow.newInstance({
        rootContainer: vtkContainerRef.current,
        containerStyle: { width: "100%", height: "70%" },
      });

      const renderer = fullScreenRenderer.getRenderer();
      const renderWindow = fullScreenRenderer.getRenderWindow();

      context.current = {
        fullScreenRenderer,
        renderWindow,
        renderer,
      };

      // Fetch STL files and initialize rendering here...

      // Add event listener for keydown events
      window.addEventListener("keydown", handleKeyDown);

      // Cleanup on unmount
      return () => {
        window.removeEventListener("keydown", handleKeyDown);
        if (context.current) {
          const { fullScreenRenderer, renderer } = context.current;
          renderer.getActors().forEach((actor) => actor.delete());
          fullScreenRenderer.delete();
          context.current = null;
        }
      };
    }
  }, []);

  return <div ref={vtkContainerRef} style={{ width: "100%", height: "100%" }} />;
};

export default VtkComponent;
