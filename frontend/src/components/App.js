import React from "react";
import SimecoApp from "./SimecoApp";
import VtkApp from "./VtkApp";

const App = () => {
  return (
    <div className="app-container">
      <div className="half-screen">
        <SimecoApp />
      </div>
      <div className="half-screen">
        <VtkApp />
      </div>
    </div>
  );
};

export default App;
