import React, { useRef, useEffect } from "react";

const WebViewSimulator = () => {
  const iframeRef = useRef(null);

  useEffect(() => {
    console.log("Antes del montaje:", iframeRef.current); // Será null
    setTimeout(() => {
      console.log("Después del montaje:", iframeRef.current); // Ahora debería ser el iframe
    }, 1000);
  }, []);

  return (
    <iframe
      ref={iframeRef}
      src="http://localhost:5173/public/chatbot/87eb897f-0365-459e-aa6b-993aae85781e"
      style={{ width: "100%", height: "500px", border: "1px solid black" }}
    />
  );
};

export default WebViewSimulator;
