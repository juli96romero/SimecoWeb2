// src/utils/websocket.js
export const initializeWebSocket = () => {
    const chatSocket = new WebSocket(`ws://${window.location.host}/ws/socket-principal-front/`);
    
    return chatSocket;
  };
  