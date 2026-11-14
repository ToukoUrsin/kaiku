// Pointer driving for the lead scene. Coordinates sent to the server are pelvis-frame x/y.
import * as THREE from '../vendor/three/three.module.min.js';

export class Drive {
  constructor({ canvas, rig, send, tableZ = 0.01, onState = () => {} }) {
    this.canvas = canvas;
    this.rig = rig;
    this.send = send;
    this.tableZ = tableZ;
    this.onState = onState;
    this.active = false;
    this.pointer = null;
    this.last = 0;
    this.pending = null;
    this.origin = new THREE.Vector3();
    this.direction = new THREE.Vector3();
    this.point = new THREE.Vector3();
    this.bind();
  }

  setActive(active) {
    this.active = Boolean(active);
    this.rig.driving = this.active;
    this.canvas.toggleAttribute('data-driving', this.active);
    this.onState(this.active);
  }

  start() {
    if (this.active) return;
    if (this.send({ type: 'cmd', name: 'drive_start' })) this.setActive(true);
  }

  stop() {
    if (!this.active) return;
    this.send({ type: 'cmd', name: 'drive_end' });
    this.setActive(false);
    this.pointer = null;
  }

  table(clientX, clientY) {
    this.rig.ray(clientX, clientY, this.origin, this.direction);
    const t = (this.tableZ - this.origin.z) / this.direction.z;
    if (!Number.isFinite(t) || t < 0) return null;
    return this.point.copy(this.origin).addScaledVector(this.direction, t);
  }

  sendPoint(clientX, clientY, grip) {
    const p = this.table(clientX, clientY);
    if (!p) return;
    const message = { type: 'cmd', name: 'hand', value: { x: p.x, y: p.y, grip } };
    const now = performance.now();
    if (now - this.last >= 30) {
      this.last = now;
      this.send(message);
    } else {
      this.pending = message;
    }
  }

  flush() {
    if (this.pending && performance.now() - this.last >= 30) {
      this.last = performance.now();
      this.send(this.pending);
      this.pending = null;
    }
    requestAnimationFrame(() => this.flush());
  }

  bind() {
    this.canvas.addEventListener('pointerdown', (event) => {
      if (!this.active) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      this.pointer = event.pointerId;
      this.canvas.setPointerCapture(event.pointerId);
      this.sendPoint(event.clientX, event.clientY, true);
    });
    this.canvas.addEventListener('pointermove', (event) => {
      if (!this.active || this.pointer !== event.pointerId) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      this.sendPoint(event.clientX, event.clientY, true);
    });
    const release = (event) => {
      if (!this.active || this.pointer !== event.pointerId) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      this.sendPoint(event.clientX, event.clientY, false);
      this.pointer = null;
    };
    this.canvas.addEventListener('pointerup', release);
    this.canvas.addEventListener('pointercancel', release);
    this.flush();
  }
}
