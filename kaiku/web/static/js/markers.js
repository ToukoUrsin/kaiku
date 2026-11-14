// The floor marker of every station: an outline that says what its robot is doing. Working
// stations answer the echo rings as they pass; a robot that sorted its scene turns green, a rank
// at a time, so the end of an episode reads as a wave.

import * as THREE from '../vendor/three/three.module.min.js';
import { RING_GLSL } from './stage.js';

export const IDLE = 0;
export const RUNNING = 1;
export const SUCCESS = 2;
export const FAILED = 3;
export const LEAD = 4;

const PAD_CENTRE_X = 0.2;             // the pad is centred between the robot and its table
const PAD_HALF = [0.66, 0.7];         // metres
const PAD_MARGIN = 0.3;               // room around the outline for its glow
const LIFT = 0.006;                   // above the floor
const RING_SPEED = 5.5;               // metres per second, as in RING_GLSL

//                 r     g     b    line  fill  ring
const LOOK = [
  /* idle    */ [0.62, 0.66, 0.74, 0.20, 0.000, 0.0],
  /* running */ [0.78, 0.82, 0.90, 0.36, 0.012, 1.0],
  /* success */ [0.29, 0.87, 0.50, 1.00, 0.150, 0.0],
  /* failed  */ [0.94, 0.41, 0.29, 0.85, 0.070, 0.0],
  /* lead    */ [1.00, 1.00, 1.00, 0.80, 0.030, 0.0],
];
const WAVE_S = 0.05;                  // seconds per metre: results sweep back through the ranks

const VERTEX = /* glsl */ `
  attribute vec4 aTint;       // rgb, fill
  attribute vec4 aLine;       // line, ring, flash, kind (1: the operator's station)
  attribute vec2 aEcho;       // distance the ring travels to get here, emphasis (hover, focus)
  uniform vec2 uHalf;
  varying vec2 vLocal;
  varying vec4 vTint;
  varying vec4 vLine;
  varying vec2 vEcho;
  varying float vDepth;
  void main() {
    vLocal = position.xy;
    vTint = aTint;
    vLine = aLine;
    vEcho = aEcho;
    vec4 view = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
    vDepth = -view.z;
    gl_Position = projectionMatrix * view;
  }
`;

const FRAGMENT = /* glsl */ `
  uniform vec2 uHalf;
  uniform float uTime;
  uniform float uRipple;
  uniform float uFogNear;
  uniform float uFogFar;
  varying vec2 vLocal;
  varying vec4 vTint;
  varying vec4 vLine;
  varying vec2 vEcho;
  varying float vDepth;
  ${RING_GLSL}

  float roundedBox(vec2 p, vec2 extent, float radius) {
    vec2 d = abs(p) - extent + radius;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - radius;
  }

  void main() {
    float sd = roundedBox(vLocal, uHalf, 0.14);
    float px = fwidth(sd);
    // never thinner than a pixel and a half, so far ranks keep their outlines
    float width = max(0.012, 1.5 * px);
    float line = 1.0 - smoothstep(width * 0.5 - px, width * 0.5 + px, abs(sd));
    if (vLine.w > 0.5) {
      float inner = abs(sd + 0.075);
      line = max(line, 0.55 * (1.0 - smoothstep(width * 0.4 - px, width * 0.4 + px, inner)));
    }
    float inside = 1.0 - smoothstep(-px, px, sd);
    float ring = uRipple * vLine.y * kaikuRing(vEcho.x, uTime);
    float flash = vLine.z;

    float fill = inside * (vTint.a + 0.5 * flash) * (0.3 + 0.7 * smoothstep(-0.6, 0.0, sd));
    float glow = (0.55 * flash + 0.10 * vTint.a * 6.0) * (1.0 - inside) * exp(-sd * 11.0);
    float alpha = line * min(1.0, vLine.x + 0.55 * ring + vEcho.y + flash) + fill + glow;

    vec3 color = mix(vTint.rgb, vec3(1.0), 0.35 * flash + 0.5 * vEcho.y);
    alpha *= 1.0 - smoothstep(uFogNear, uFogFar, vDepth);
    gl_FragColor = vec4(color, clamp(alpha, 0.0, 1.0));
  }
`;

export class Markers {
  constructor(stage, layout, nEnvs, groundZ) {
    this.stage = stage;
    this.n = nEnvs;
    const geometry = new THREE.PlaneGeometry(2 * (PAD_HALF[0] + PAD_MARGIN), 2 * (PAD_HALF[1] + PAD_MARGIN));
    this.tint = new THREE.InstancedBufferAttribute(new Float32Array(nEnvs * 4), 4);
    this.line = new THREE.InstancedBufferAttribute(new Float32Array(nEnvs * 4), 4);
    this.echo = new THREE.InstancedBufferAttribute(new Float32Array(nEnvs * 2), 2);
    for (const attribute of [this.tint, this.line, this.echo]) attribute.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('aTint', this.tint);
    geometry.setAttribute('aLine', this.line);
    geometry.setAttribute('aEcho', this.echo);

    this.uniforms = {
      uHalf: { value: new THREE.Vector2(PAD_HALF[0], PAD_HALF[1]) },
      uFogNear: { value: 8 },
      uFogFar: { value: 34 },
      ...stage.ring,
    };
    const material = new THREE.ShaderMaterial({
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      uniforms: this.uniforms,
      transparent: true,
      depthWrite: false,
    });
    const mesh = new THREE.InstancedMesh(geometry, material, nEnvs);
    mesh.frustumCulled = false;
    mesh.renderOrder = 1;
    const matrix = new THREE.Matrix4();
    this.reach = new Float32Array(nEnvs);         // metres from the operator's robot
    for (let e = 0; e < nEnvs; e++) {
      const x = layout.offsets[e * 3], y = layout.offsets[e * 3 + 1], z = layout.offsets[e * 3 + 2];
      matrix.makeTranslation(x + PAD_CENTRE_X, y, z + groundZ + LIFT);
      mesh.setMatrixAt(e, matrix);
      this.reach[e] = Math.hypot(x, y);
    }
    stage.scene.add(mesh);
    this.mesh = mesh;

    this.shown = new Uint8Array(nEnvs).fill(255);     // state each marker is easing towards
    this.wanted = new Uint8Array(nEnvs).fill(255);    // state the stream reports
    this.changeAt = new Float64Array(nEnvs);          // when `wanted` takes effect
    this.flash = new Float32Array(nEnvs);
    this.emphasis = new Float32Array(nEnvs);
    this.hovered = -1;
    this.focused = -1;
  }

  /**
   * @param {Uint8Array} states  state per robot from the stream
   * @param {number[]|null} lag  seconds each robot runs behind the operator's
   */
  update(states, lag, seconds, dt) {
    const tint = this.tint.array, line = this.line.array, echo = this.echo.array;
    const ease = 1 - Math.exp(-dt * 9);
    for (let e = 0; e < this.n; e++) {
      const state = states[e] <= LEAD ? states[e] : IDLE;
      if (state !== this.wanted[e]) {
        this.wanted[e] = state;
        const settled = state === SUCCESS || state === FAILED;
        this.changeAt[e] = this.shown[e] === 255 ? 0 : seconds + (settled ? WAVE_S : 0.012) * this.reach[e];
      }
      if (this.shown[e] !== this.wanted[e] && seconds >= this.changeAt[e]) {
        const first = this.shown[e] === 255;
        this.shown[e] = this.wanted[e];
        if (this.shown[e] === SUCCESS && !first) this.flash[e] = 1;
        if (first) this.snap(e);
      }
      const look = LOOK[this.shown[e] === 255 ? IDLE : this.shown[e]];
      const t = e * 4;
      tint[t] += (look[0] - tint[t]) * ease;
      tint[t + 1] += (look[1] - tint[t + 1]) * ease;
      tint[t + 2] += (look[2] - tint[t + 2]) * ease;
      tint[t + 3] += (look[4] - tint[t + 3]) * ease;
      line[t] += (look[3] - line[t]) * ease;
      line[t + 1] += (look[5] - line[t + 1]) * ease;
      this.flash[e] *= Math.exp(-dt * 2.4);
      line[t + 2] = this.flash[e];
      line[t + 3] = this.shown[e] === LEAD ? 1 : 0;

      // a robot that runs behind answers the ring that much later
      echo[e * 2] = this.reach[e] + RING_SPEED * (lag ? lag[e] || 0 : 0);
      const emphasis = e === this.focused ? 0.55 : e === this.hovered ? 0.35 : 0;
      this.emphasis[e] += (emphasis - this.emphasis[e]) * (1 - Math.exp(-dt * 14));
      echo[e * 2 + 1] = this.emphasis[e];
    }
    this.tint.needsUpdate = true;
    this.line.needsUpdate = true;
    this.echo.needsUpdate = true;
    this.uniforms.uFogNear.value = this.stage.scene.fog.near;
    this.uniforms.uFogFar.value = this.stage.scene.fog.far;
  }

  snap(e) {
    const look = LOOK[this.shown[e]];
    const t = e * 4;
    this.tint.array.set([look[0], look[1], look[2], look[4]], t);
    this.line.array[t] = look[3];
    this.line.array[t + 1] = look[5];
  }
}
