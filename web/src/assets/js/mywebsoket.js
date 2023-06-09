// websocket.js

class WebSocketModule {
  constructor(wsuri, onMessageCallback) {
    this.wsuri = wsuri;
    this.onMessageCallback = onMessageCallback;
    this.websock = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
    this.reconnectInterval = 1000;
  }

  connect() {
    this.websock = new WebSocket(this.wsuri);

    this.websock.onopen = () => {
      this.isConnected = true;
      console.log('WebSocket connected');
    };

    this.websock.onclose = () => {
      this.isConnected = false;
      console.log('WebSocket disconnected');

      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        setTimeout(() => this.connect(), this.reconnectInterval);
      } else {
        console.error('Max reconnect attempts reached');
      }
    };

    this.websock.onmessage = this.onMessageCallback;

    this.websock.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  send(data) {
    if (this.websock && this.isConnected) {
      this.websock.send(data);
    } else {
      console.error('WebSocket is not connected');
    }
  }

  close() {
    if (this.websock && this.isConnected) {
      this.websock.close();
    }
  }
}

export {WebSocketModule};
