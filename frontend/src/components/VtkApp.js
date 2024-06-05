import React, { useState, useRef, useEffect } from 'react';
import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkSTLReader from '@kitware/vtk.js/IO/Geometry/STLReader'; // Import STL reader instead of OBJ reader

const VtkApp = () => {
  const vtkContainerRef = useRef(null);
  const context = useRef(null);
  const [representation, setRepresentation] = useState(2);

  useEffect(() => {
    const fetchStlFiles = async () => { // Change the function name to fetchStlFiles
      try {
        const response = await fetch('/api/stl-files/'); // Fetch STL files instead of OBJ files
        const files = await response.json();
        return files;
      } catch (error) {
        console.error('Error fetching STL files:', error);
        return [];
      }
    };

    if (!context.current && vtkContainerRef.current) {
      console.log('Initializing VTK...');
      const fullScreenRenderer = vtkFullScreenRenderWindow.newInstance({
        rootContainer: vtkContainerRef.current,
        containerStyle: { width: '130%', height: '130%' }
      });

      const renderer = fullScreenRenderer.getRenderer();
      const renderWindow = fullScreenRenderer.getRenderWindow();

      context.current = {
        fullScreenRenderer,
        renderWindow,
        renderer,
      };

      // Load and display all STL files from the server
      fetchStlFiles().then(stlFiles => { // Change function call to fetchStlFiles
        stlFiles.forEach(file => {
          const reader = vtkSTLReader.newInstance(); // Use STL reader instead of OBJ reader
          reader.setUrl(`/static/${file}`).then(() => {
            console.log(`Successfully loaded: ${file}`);
            const source = reader.getOutputData(0);
            const mapper = vtkMapper.newInstance();
            mapper.setInputData(source);

            const actor = vtkActor.newInstance();
            actor.setMapper(mapper);
            

            renderer.addActor(actor);
            
            renderer.resetCamera();
            renderWindow.render();
            console.log(`Rendered: ${file}`);
          }).catch(error => {
            console.error(`Error loading ${file}:`, error);
          });
        });
      });
    }

    return () => {
      if (context.current) {
        console.log('Cleaning up VTK...');
        const { fullScreenRenderer, renderer } = context.current;
        renderer.getActors().forEach(actor => actor.delete());
        fullScreenRenderer.delete();
        context.current = null;
      }
    };
  }, [vtkContainerRef]);

  useEffect(() => {
    if (context.current) {
      console.log('Updating representation...');
      const { renderer, renderWindow } = context.current;
      renderer.getActors().forEach(actor => {
        actor.getProperty().setRepresentation(representation);
      });
      renderWindow.render();
    }
  }, [representation]);

  return (
    <div>
      <div ref={vtkContainerRef} className="vtk-container" />
      <table className="control-table">
        <tbody>
          <tr>
            <td>
              <select
                value={representation}
                style={{ width: '100%' }}
                onChange={(ev) => setRepresentation(Number(ev.target.value))}
              >
                <option value="0">Points</option>
                <option value="1">Wireframe</option>
                <option value="2">Surface</option>
              </select>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default VtkApp;
