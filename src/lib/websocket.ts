/**
 * WebSocket client for real-time fact-check events.
 *
 * Usage:
 *   const ws = new ResearchWebSocket(sessionId);
 *   ws.onEvent((event) => console.log(event));
 *   ws.connect();
 */

export interface AgentEvent {
  session_id: string;
  event_type: string;
  agent: string;
  timestamp: string;
  data: Record<string, any>;
}

export type EventCallback = (event: AgentEvent) => void;

export type ConnectionStatus = "connecting" | "connected" | "reconnecting" | "disconnected";
export type StatusCallback = (status: ConnectionStatus, detail?: string) => void;

export class ResearchWebSocket {
  private ws: WebSocket | null = null;
  private sessionId: string;
  private callbacks: Set<EventCallback> = new Set();
  private statusCallbacks: Set<StatusCallback> = new Set();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private pingInterval: NodeJS.Timeout | null = null;
  private intentionalDisconnect = false;
  private _status: ConnectionStatus = "disconnected";

  constructor(sessionId: string) {
    this.sessionId = sessionId;
  }

  get status(): ConnectionStatus {
    return this._status;
  }

  private setStatus(status: ConnectionStatus, detail?: string): void {
    this._status = status;
    this.statusCallbacks.forEach((cb) => {
      try { cb(status, detail); } catch { /* ignore */ }
    });
  }

  onStatusChange(callback: StatusCallback): () => void {
    this.statusCallbacks.add(callback);
    return () => { this.statusCallbacks.delete(callback); };
  }

  connect(): void {
    const url = `ws://localhost:9090/ws/${this.sessionId}`;
    this.intentionalDisconnect = false;
    this.setStatus(this.reconnectAttempts > 0 ? "reconnecting" : "connecting",
      this.reconnectAttempts > 0 ? `Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}` : undefined);

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.setStatus("connected");

        // Start ping/pong to keep connection alive
        this.startPing();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Handle system messages
          if (data.type === "connected") {
            return;
          }

          if (data.type === "pong") {
            // Keepalive response
            return;
          }

          // Handle agent events
          if (data.event_type) {
            this.notifyCallbacks(data as AgentEvent);
          }
        } catch (error) {
          console.error("[WebSocket] Failed to parse message:", error);
        }
      };

      this.ws.onerror = (event) => {
        // Avoid noisy empty error objects in Next.js overlay
        const detail = (event as any)?.message;
        if (detail) {
          console.warn("[WebSocket] Error:", detail);
        } else {
          console.warn("[WebSocket] Error event");
        }
      };

      this.ws.onclose = () => {
        this.stopPing();

        if (this.intentionalDisconnect) {
          this.setStatus("disconnected");
          return;
        }

        // Attempt reconnection
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          const delay = this.reconnectDelay * this.reconnectAttempts;
          this.setStatus("reconnecting", `Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${Math.round(delay / 1000)}s`);
          setTimeout(() => this.connect(), delay);
        } else {
          this.setStatus("disconnected", "Max reconnection attempts reached");
          console.error("[WebSocket] Max reconnection attempts reached");
        }
      };
    } catch (error) {
      console.error("[WebSocket] Failed to create connection:", error);
    }
  }

  disconnect(): void {
    this.intentionalDisconnect = true;
    this.stopPing();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  onEvent(callback: EventCallback): () => void {
    this.callbacks.add(callback);

    // Return unsubscribe function
    return () => {
      this.callbacks.delete(callback);
    };
  }

  private notifyCallbacks(event: AgentEvent): void {
    this.callbacks.forEach((callback) => {
      try {
        callback(event);
      } catch (error) {
        console.error("[WebSocket] Callback error:", error);
      }
    });
  }

  private startPing(): void {
    // Send ping every 30 seconds
    this.pingInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send("ping");
      }
    }, 30000);
  }

  private stopPing(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
