// src/components/App.js
import React from 'react';
import TryVTK from './TryVTK';


const App = () => {
  return (
    <div className="app-container">
      <TryVTK/>
      {console.log("Renderizando App.js")}
    </div>
  );
};

export default App;
