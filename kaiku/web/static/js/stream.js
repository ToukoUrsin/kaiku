// The live connection: joint positions of every robot (binary, about 30 Hz), the HUD (JSON, 10 Hz)
// and one-off events. Reconnects on its own and reports what it is doing through `onStatus`.

import { applyFrame, ackMessage, KIND_DELTA } from './protocol.js';

const RETRY_MS = [400, 800, 1500, 2500, 4000];
const STALL_MS = 2500;

export class Stream {
  /**
   * @param {object} opts
   * @param {number} opts.nEnvs
   * @param {number} opts.nq
   * @param {string} opts.modelHash  hash of the scene the page has loaded
   */
  constructor({ nEnvs, nq, modelHash }) {
    this.nEnvs = nEnvs;
    this.nq = nq;
    this.modelHash = modelHash;

    // The two newest frames; the stage draws a blend of them so 30 Hz data moves at 60 fps.
    this.prev = new Int16Array(nEnvs * nq);
    this.cur = new Int16Array(nEnvs * nq);
    this.state = new Uint8Array(nEnvs);
    this.frames = 0;            // frames decoded since the page loaded
    this.arrivedAt = 0;         // performance.now() of the newest frame
    this.interval = 33;         // smoothed milliseconds between frames
    this.seq = -1;

    this.status = 'connecting'; // connecting | live | reconnecting
    this.retryAt = 0;
    this.attempt = 0;
    this.onStatus = () => {};
    this.onHud = () => {};
    this.onEvent = () => {};
    this.onModelChanged = () => {};

    this.socket = null;
    this.timer = 0;
    this.closed = false;
    this.hasFrame = false;

    document.addEventListener('visibilitychange', () => {
      this.sendControl({ _: 'pause', on: document.hidden });
    });
  }

  connect() {
    clearTimeout(this.timer);
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    let socket;
    try {
      socket = new WebSocket(`${scheme}://${location.host}/api/ws`);
    } catch {
      this.scheduleRetry();
      return;
    }
    socket.binaryType = 'arraybuffer';
    this.socket = socket;
    this.hasFrame = false;

    socket.onmessage = (event) => {
      if (socket !== this.socket) return;
      if (typeof event.data === 'string') this.onText(event.data);
      else this.onBinary(event.data);
    };
    socket.onclose = () => {
      if (socket !== this.socket) return;
      this.socket = null;
      if (!this.closed) this.scheduleRetry();
    };
    // An error is always followed by a close; that is where the retry starts.
    socket.onerror = () => {};
  }

  close() {
    this.closed = true;
    clearTimeout(this.timer);
    this.socket?.close();
  }

  /** A command for the runner, e.g. `{type: 'cmd', name: 'pause', value: true}`. */
  send(message) {
    if (this.socket?.readyState !== WebSocket.OPEN) return false;
    this.socket.send(JSON.stringify(message));
    return true;
  }

  sendControl(message) {
    this.send(message);
  }

  /** How far the stage is between `prev` (0) and `cur` (1) at time `now`. */
  blend(now) {
    if (this.frames < 2) return 1;
    const t = (now - this.arrivedAt) / this.interval;
    return t < 0 ? 0 : t > 1 ? 1 : t;
  }

  /** True while frames keep arriving. */
  flowing(now) {
    return this.status === 'live' && now - this.arrivedAt < STALL_MS;
  }

  setStatus(status) {
    if (status === this.status) return;
    this.status = status;
    this.onStatus(status);
  }

  scheduleRetry() {
    const wait = RETRY_MS[Math.min(this.attempt, RETRY_MS.length - 1)];
    this.attempt += 1;
    this.retryAt = performance.now() + wait;
    this.setStatus('reconnecting');
    this.timer = setTimeout(() => this.connect(), wait);
  }

  onText(text) {
    let message;
    try {
      message = JSON.parse(text);
    } catch {
      return;
    }
    if (message._ === 'hud') {
      this.onHud(message.d || {});
    } else if (message._ === 'event') {
      this.onEvent(message.d || {});
    } else if (message._ === 'hello') {
      if (message.model_hash && message.model_hash !== this.modelHash) this.onModelChanged();
      if (document.hidden) this.sendControl({ _: 'pause', on: true });
    }
  }

  onBinary(buffer) {
    const bytes = new Uint8Array(buffer);
    // A delta only makes sense on top of the frame it was made against.
    const isDelta = bytes[1] === KIND_DELTA;
    if (isDelta) {
      const base = new DataView(buffer).getUint32(8, true);
      if (!this.hasFrame || base !== this.seq) {
        this.sendControl({ _: 'key' });
        return;
      }
    }
    this.prev.set(this.cur);
    let head;
    try {
      head = applyFrame(bytes, this.cur, this.state);
    } catch {
      this.hasFrame = false;
      this.sendControl({ _: 'key' });
      return;
    }
    if (!this.hasFrame) this.prev.set(this.cur);   // first frame after a gap: nothing to blend from
    this.hasFrame = true;
    this.seq = head.seq;
    this.socket.send(ackMessage(head.seq));

    const now = performance.now();
    if (this.frames > 0) {
      const gap = Math.min(120, Math.max(12, now - this.arrivedAt));
      this.interval += 0.1 * (gap - this.interval);
    }
    this.arrivedAt = now;
    this.frames += 1;
    this.attempt = 0;
    this.setStatus('live');
  }
}
