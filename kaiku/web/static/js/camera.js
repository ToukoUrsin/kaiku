// The camera: three named shots, a close-up of any one robot, and free orbiting. The camera is
// always described the same way (what it looks at, from which side, from how high, from how far)
// and eases towards that description, so a shot change and a drag are the same motion.

import * as THREE from '../vendor/three/three.module.min.js';

const FOV = 32;
const TARGET_X = 0, TARGET_Y = 1, TARGET_Z = 2, AZIMUTH = 3, POLAR = 4, DISTANCE = 5;
const MIN_POLAR = 0.02;
const MAX_POLAR = 1.52;
const MIN_DISTANCE = 1.1;
const MAX_DISTANCE = 70;
const DEG = Math.PI / 180;

export class CameraRig {
  /**
   * @param {import('./stage.js').Stage} stage
   * @param {HTMLElement} surface  element that receives the pointer
   * @param {ReturnType<import('./layout.js').formation>} layout
   */
  constructor(stage, surface, layout) {
    this.stage = stage;
    this.camera = stage.camera;
    this.camera.fov = FOV;
    this.surface = surface;
    this.offsets = layout.offsets;

    const depth = layout.depth;
    const width = Math.max(layout.width, 4);
    // `frame` is the width and height (metres) the shot must show; the distance follows from it,
    // so a shot holds its subject on a wide laptop and on a tall phone alike.
    this.shots = {
      lead: { target: [0.25, 0.0, 0.38], azimuth: -30 * DEG, polar: 76 * DEG, frame: [2.1, 1.9] },
      overview: {
        target: [-0.5 * depth, 0, 0.45 + 0.05 * depth], azimuth: -12 * DEG, polar: 64 * DEG,
        frame: [width * 0.95 + 1.0, depth * 0.5 + 2.6],
      },
      top: { target: [-0.5 * depth + 0.4, 0, 0], azimuth: 0, polar: MIN_POLAR, frame: [width + 2.8, depth + 3.6] },
    };
    // the opening starts here: the operator's hands and the robot's hands under them, nothing else
    this.hands = { target: [0.3, 0.0, 0.62], azimuth: -38 * DEG, polar: 70 * DEG, frame: [1.25, 1.05] };

    this.current = new Float64Array(6);
    this.goal = new Float64Array(6);
    this.velocity = new Float64Array(6);
    this.shot = 'lead';
    this.focus = -1;
    this.orbit = false;
    this.settle = 0.55;           // seconds the camera takes to arrive
    this.shift = new THREE.Vector2(0, 0);   // where the subject sits on screen, in half-screens from the centre
    this.shiftGoal = new THREE.Vector2(0, 0);
    this.started = false;

    this.pointers = new Map();
    this.dragged = 0;
    this.pinch = 0;
    this.onPick = () => {};
    this.onHover = () => {};
    this.onChange = () => {};
    this.scratch = new THREE.Vector3();
    this.bind();
  }

  /** Go to a named shot: 'lead', 'overview' or 'top'. `settle` is the seconds to take over it. */
  goTo(name, settle) {
    const shot = this.shots[name];
    if (!shot) return;
    this.shot = name;
    this.focus = -1;
    this.aim(shot, 0, 0, 0);
    if (settle) this.settle = settle;
    this.onChange();
  }

  /** Look at something that is not one of the named shots (the page's opening does). */
  look(shot, settle) {
    this.shot = 'opening';
    this.focus = -1;
    this.aim(shot, 0, 0, 0);
    if (settle) this.settle = settle;
    this.onChange();
  }

  /** Close-up of one robot's station. */
  focusOn(env) {
    this.shot = 'focus';
    this.focus = env;
    const o = this.offsets;
    this.aim(this.shots.lead, o[env * 3], o[env * 3 + 1], o[env * 3 + 2]);
    this.onChange();
  }

  aim(shot, dx, dy, dz) {
    const goal = this.goal;
    goal[TARGET_X] = shot.target[0] + dx;
    goal[TARGET_Y] = shot.target[1] + dy;
    goal[TARGET_Z] = shot.target[2] + dz;
    // turn the short way round
    goal[AZIMUTH] = this.current[AZIMUTH] + wrap(shot.azimuth - this.current[AZIMUTH]);
    goal[POLAR] = shot.polar;
    goal[DISTANCE] = this.fit(shot.frame);
    this.frame = shot.frame;
    this.settle = 0.55;
    if (!this.started) {
      this.current.set(goal);
      this.started = true;
    }
  }

  fit(frame) {
    const tan = Math.tan((FOV * DEG) / 2);
    const aspect = this.camera.aspect || 1;
    return Math.min(MAX_DISTANCE, Math.max(frame[1] / (2 * tan), frame[0] / (2 * tan * aspect)));
  }

  /** The stage was resized: keep the current shot's subject in view. */
  refit() {
    if (this.shot !== 'free' && this.frame) this.goal[DISTANCE] = this.fit(this.frame);
  }

  get distance() {
    return this.current[DISTANCE];
  }

  update(dt) {
    if (this.orbit && this.pointers.size === 0) this.goal[AZIMUTH] += 0.07 * dt;
    for (let i = 0; i < 6; i++) this.ease(i, dt);
    this.shift.x += (this.shiftGoal.x - this.shift.x) * (1 - Math.exp(-dt * 5));
    this.shift.y += (this.shiftGoal.y - this.shift.y) * (1 - Math.exp(-dt * 5));

    const c = this.current;
    const sin = Math.sin(c[POLAR]);
    const camera = this.camera;
    camera.position.set(
      c[TARGET_X] + c[DISTANCE] * sin * Math.cos(c[AZIMUTH]),
      c[TARGET_Y] + c[DISTANCE] * sin * Math.sin(c[AZIMUTH]),
      c[TARGET_Z] + c[DISTANCE] * Math.cos(c[POLAR]),
    );
    this.scratch.set(c[TARGET_X], c[TARGET_Y], c[TARGET_Z]);
    camera.lookAt(this.scratch);
    camera.near = Math.max(0.05, c[DISTANCE] * 0.04);
    camera.far = c[DISTANCE] + 110;
    camera.updateProjectionMatrix();
    // slide the picture so the subject clears the panels laid over the stage
    camera.projectionMatrix.elements[8] = -this.shift.x;
    camera.projectionMatrix.elements[9] = -this.shift.y;
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    camera.updateMatrixWorld();
  }

  // Critically damped: starts and stops gently however far the goal is.
  ease(i, dt) {
    const omega = 2 / this.settle;
    const x = omega * dt;
    const decay = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
    const change = this.current[i] - this.goal[i];
    const step = (this.velocity[i] + omega * change) * dt;
    this.velocity[i] = (this.velocity[i] - omega * step) * decay;
    this.current[i] = this.goal[i] + (change + step) * decay;
  }

  /** What the camera looks at, for the light that follows it. */
  target(out) {
    return out.set(this.current[TARGET_X], this.current[TARGET_Y], this.current[TARGET_Z]);
  }

  /** The ray under a point of the page. */
  ray(clientX, clientY, origin, direction) {
    const box = this.surface.getBoundingClientRect();
    direction.set(((clientX - box.left) / box.width) * 2 - 1, 1 - ((clientY - box.top) / box.height) * 2, 0.5);
    direction.unproject(this.camera);
    origin.copy(this.camera.position);
    direction.sub(origin).normalize();
  }

  free() {
    if (this.shot === 'free') return;
    this.shot = 'free';
    this.onChange();
  }

  bind() {
    const surface = this.surface;
    surface.addEventListener('pointerdown', (event) => {
      surface.setPointerCapture(event.pointerId);
      this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (this.pointers.size === 1) this.dragged = 0;
      this.pinch = this.spread();
    });

    surface.addEventListener('pointermove', (event) => {
      const pointer = this.pointers.get(event.pointerId);
      if (!pointer) {
        if (event.pointerType === 'mouse') this.onHover(event.clientX, event.clientY);
        return;
      }
      const dx = event.clientX - pointer.x;
      const dy = event.clientY - pointer.y;
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      this.dragged += Math.abs(dx) + Math.abs(dy);
      if (this.dragged < 5) return;
      this.settle = 0.12;

      if (this.pointers.size >= 2) {
        const spread = this.spread();
        if (this.pinch > 0 && spread > 0) this.zoom(this.pinch / spread);
        this.pinch = spread;
      } else if (event.shiftKey || event.buttons === 2) {
        this.pan(dx, dy);
      } else {
        this.goal[AZIMUTH] -= dx * 0.0055;
        this.goal[POLAR] = clamp(this.goal[POLAR] - dy * 0.0045, MIN_POLAR, MAX_POLAR);
      }
      // a robot in close-up stays the subject while the camera is turned around it
      if (this.shot !== 'focus') this.free();
    });

    const release = (event) => {
      if (!this.pointers.delete(event.pointerId)) return;
      this.pinch = this.spread();
      if (this.pointers.size === 0 && this.dragged < 5 && event.type === 'pointerup') this.onPick(event.clientX, event.clientY);
    };
    surface.addEventListener('pointerup', release);
    surface.addEventListener('pointercancel', release);
    surface.addEventListener('pointerleave', () => this.onHover(-1, -1));
    surface.addEventListener('contextmenu', (event) => event.preventDefault());

    surface.addEventListener('wheel', (event) => {
      event.preventDefault();
      this.settle = 0.16;
      this.zoom(Math.exp(clamp(event.deltaY, -240, 240) * 0.0014));
      if (this.shot !== 'focus') this.free();
    }, { passive: false });
  }

  spread() {
    if (this.pointers.size < 2) return 0;
    const [a, b] = [...this.pointers.values()];
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  zoom(factor) {
    this.goal[DISTANCE] = clamp(this.goal[DISTANCE] * factor, MIN_DISTANCE, MAX_DISTANCE);
  }

  // Slide what the camera looks at across the floor.
  pan(dx, dy) {
    const metres = (2 * this.current[DISTANCE] * Math.tan((FOV * DEG) / 2)) / this.surface.clientHeight;
    const azimuth = this.current[AZIMUTH];
    const rightX = -Math.sin(azimuth), rightY = Math.cos(azimuth);
    const forwardX = -Math.cos(azimuth), forwardY = -Math.sin(azimuth);
    this.goal[TARGET_X] += (-dx * rightX + dy * forwardX) * metres;
    this.goal[TARGET_Y] += (-dx * rightY + dy * forwardY) * metres;
  }
}

function clamp(value, low, high) {
  return value < low ? low : value > high ? high : value;
}

function wrap(angle) {
  return angle - 2 * Math.PI * Math.round(angle / (2 * Math.PI));
}
