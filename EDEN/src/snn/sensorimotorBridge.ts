/**
 * EDEN ↔ Python sensorimotor protocol bridge (client side).
 *
 * Connects to the Python core WebSocket (`serve_runtime` / port 8766 by default)
 * and translates EDEN `SnnTraceEvent` batches into protocol messages (and back).
 *
 * Wire format: one JSON object per WebSocket frame.
 * - Inbound from EDEN UI: `{ kind, value, meta, ts }` or protocol `{ type, id, payload }`
 * - Outbound from Python: protocol messages (action / global_signal / trace)
 */

export type EdenTraceEvent = {
  kind: string;
  value?: number;
  meta?: Record<string, unknown>;
  ts?: number;
  id?: string;
};

export type SensorimotorMessage = {
  type: 'register' | 'deregister' | 'observation' | 'action' | 'global_signal' | 'trace';
  id: string;
  ts: number;
  payload: Record<string, unknown>;
};

export type BridgeHandlers = {
  onMessage?: (message: SensorimotorMessage) => void;
  onEdenEvent?: (event: EdenTraceEvent) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
};

const num = (meta: Record<string, unknown> | undefined, key: string, fallback = 0): number => {
  const value = meta?.[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
};

/** Convert EDEN body/global/spike events to protocol messages (mirrors Python). */
export const edenEventsToMessages = (
  events: EdenTraceEvent[],
  moduleId = 'eden-body',
): SensorimotorMessage[] => {
  const now = Date.now() / 1000;
  return events.map((event) => {
    const ts = event.ts ?? now;
    if (event.kind === 'body') {
      const meta = event.meta ?? {};
      return {
        type: 'observation',
        id: moduleId,
        ts,
        payload: {
          values: [
            num(meta, 'overload'),
            num(meta, 'ambient_stimulus'),
            num(meta, 'nearest_stimulus'),
            num(meta, 'deformation'),
            num(meta, 'gait_drive'),
            event.value ?? 0,
          ],
          modality: 'eden_body',
          source: 'eden',
          meta,
        },
      };
    }
    if (event.kind === 'global_signal') {
      return {
        type: 'global_signal',
        id: moduleId,
        ts,
        payload: {
          reward: event.value ?? 0,
          arousal: num(event.meta, 'arousal', event.value ?? 0),
          novelty: num(event.meta, 'novelty'),
          fatigue: num(event.meta, 'fatigue'),
          source: 'eden',
        },
      };
    }
    return {
      type: 'trace',
      id: moduleId,
      ts,
      payload: {
        kind: event.kind,
        value: event.value ?? 0,
        meta: event.meta ?? {},
        source: 'eden',
      },
    };
  });
};

export const messageToEdenEvent = (message: SensorimotorMessage): EdenTraceEvent => {
  if (message.type === 'action') {
    const values = (message.payload.values as number[] | undefined)
      ?? (message.payload.command as number[] | undefined)
      ?? [];
    return {
      kind: 'action',
      id: message.id,
      ts: message.ts,
      value: values[0] ?? 0,
      meta: { values, source: 'python-core' },
    };
  }
  if (message.type === 'global_signal') {
    return {
      kind: 'global_signal',
      id: message.id,
      ts: message.ts,
      value: Number(message.payload.reward ?? message.payload.intrinsic_reward ?? 0),
      meta: { ...message.payload },
    };
  }
  if (message.type === 'observation') {
    const values = (message.payload.values as number[] | undefined) ?? [];
    return {
      kind: 'body',
      id: message.id,
      ts: message.ts,
      value: values[0] ?? 0,
      meta: { values, modality: message.payload.modality },
    };
  }
  return {
    kind: String(message.payload.kind ?? message.type),
    id: message.id,
    ts: message.ts,
    value: Number(message.payload.value ?? 0),
    meta: (message.payload.meta as Record<string, unknown> | undefined) ?? message.payload,
  };
};

export const makeRegisterMessage = (moduleId = 'eden-body'): SensorimotorMessage => ({
  type: 'register',
  id: moduleId,
  ts: Date.now() / 1000,
  payload: {
    role: 'both',
    modality: 'eden_body',
    shape: [6],
    action_space: { type: 'continuous', dims: 4, range: [-1, 1] },
  },
});

/**
 * Browser WebSocket client for the Python sensorimotor runtime.
 *
 * ```ts
 * const bridge = new SensorimotorBridge('ws://127.0.0.1:8766');
 * bridge.connect();
 * bridge.sendEvents([{ kind: 'body', meta: { gait_drive: 0.5 } }]);
 * ```
 */
export class SensorimotorBridge {
  private socket: WebSocket | null = null;
  private readonly url: string;
  private readonly moduleId: string;
  private readonly handlers: BridgeHandlers;

  constructor(url = 'ws://127.0.0.1:8766', moduleId = 'eden-body', handlers: BridgeHandlers = {}) {
    this.url = url;
    this.moduleId = moduleId;
    this.handlers = handlers;
  }

  connect(): void {
    if (typeof WebSocket === 'undefined') {
      throw new Error('WebSocket is not available in this environment');
    }
    this.socket = new WebSocket(this.url);
    this.socket.onopen = () => {
      this.send(makeRegisterMessage(this.moduleId));
      this.handlers.onOpen?.();
    };
    this.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(String(event.data)) as SensorimotorMessage;
        this.handlers.onMessage?.(data);
        this.handlers.onEdenEvent?.(messageToEdenEvent(data));
      } catch {
        // ignore malformed frames
      }
    };
    this.socket.onclose = () => this.handlers.onClose?.();
    this.socket.onerror = (error) => this.handlers.onError?.(error);
  }

  send(message: SensorimotorMessage): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    }
  }

  sendEvents(events: EdenTraceEvent[]): void {
    for (const message of edenEventsToMessages(events, this.moduleId)) {
      this.send(message);
    }
  }

  close(): void {
    this.socket?.close();
    this.socket = null;
  }

  get connected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }
}
