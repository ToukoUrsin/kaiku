// A visitor drives the lead robot: with a pointer on the table, or with a hand in front of the
// camera. The page says where the hand is and whether it pinches; with a camera it also sends the
// hand's 21 landmarks in metres and how near the hand is, and the runner turns those into the robot's
// fingers the way it does for gloves. The runner decides whose turn it is, so everything here follows
// what it last said.

import * as THREE from '../vendor/three/three.module.min.js';

const SEND_MS = 33;                   // as often as the robots are commanded
const NEAR_M = 0.05;                  // a part this close to the pointer is the one a press takes
const PINCH_ON = 0.38;                // thumb tip to index tip, in hand lengths: closer is a pinch
const PINCH_OFF = 0.55;               // further is open again
const HAND_LOST_MS = 600;             // no hand in the picture for this long: it is not seen (one missed frame is not that)
const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
const MEDIAPIPE = new URL('../vendor/mediapipe/', import.meta.url).href;

export class Drive {
  constructor({ stage, canvas, rig, send, tableZ = 0.01, onState = () => {}, onNote = () => {} }) {
    this.canvas = canvas;
    this.rig = rig;
    this.send = send;
    this.tableZ = tableZ;
    this.onState = onState;
    this.onNote = onNote;             // something the visitor should read: the camera would not start
    this.active = false;              // the runner says this page is driving
    this.asked = 0;                   // when this page last asked for a turn
    this.table = null;                // the parts and bins, as the runner last told them
    this.want = null;                 // the bin the part in the hand belongs in
    this.at = null;                   // where the hand is asked to be: {x, y} on the table
    this.grip = false;
    this.pointer = null;
    this.last = 0;
    this.dirty = false;
    this.camera = null;               // {video, stream, landmarker, canvas} while a camera hand drives
    this.own = false;                 // the visitor does the grasp themselves: nothing scripted
    this.fingers = null;              // whose fingers the runner has: 'camera', 'glove' or null
    this.glove = false;               // the visitor wears a glove: fingers without a camera
    this.origin = new THREE.Vector3();
    this.direction = new THREE.Vector3();
    this.point = new THREE.Vector3();
    this.marks = new Marks(stage, tableZ);
    this.bind();
  }

  /** Ask the runner for a turn. Nothing changes here until it says yes. */
  start() {
    if (this.active) return;
    this.asked = performance.now();
    this.send({ type: 'cmd', name: 'drive_start' });
  }

  stop() {
    this.send({ type: 'cmd', name: 'drive_end' });
    this.asked = 0;
    this.follow(false, null);
  }

  /** What the runner last said: is this page driving, and what is on the table. */
  follow(mine, operator) {
    this.table = mine ? operator?.table || null : null;
    this.want = mine ? operator?.want || null : null;
    this.fingers = mine ? operator?.fingers || null : null;
    this.glove = mine && Boolean(operator?.glove);
    if (mine === this.active) return;
    this.active = mine;
    this.rig.driving = mine;
    this.canvas.toggleAttribute('data-driving', mine);
    if (!mine) {
      this.pointer = null;
      this.at = null;
      this.grip = false;
      this.own = false;
      this.stopCamera();
    } else if (operator?.glove?.coil) {
      // a glove on a Coil Pro says where the hand is and does the grasp: the page only keeps the turn
      this.own = true;
      this.at = this.at || { x: 0.28, y: 0 };
    }
    this.onState(mine);
  }

  /** The camera is on and has not seen a hand for a moment (or not yet at all). */
  get handLost() {
    return Boolean(this.camera) && performance.now() - this.camera.seen > HAND_LOST_MS;
  }

  /** Still waiting for the runner's yes. */
  get waiting() {
    return !this.active && this.asked > 0 && performance.now() - this.asked < 4000;
  }

  onTable(clientX, clientY) {
    this.rig.ray(clientX, clientY, this.origin, this.direction);
    const t = (this.tableZ - this.origin.z) / this.direction.z;
    if (!Number.isFinite(t) || t < 0) return null;
    return this.point.copy(this.origin).addScaledVector(this.direction, t);
  }

  aim(clientX, clientY, grip) {
    const p = this.onTable(clientX, clientY);
    if (!p) return;
    const r = this.table?.reach;                // the runner keeps the hand in here: show it where it will be
    this.at = r ? { x: Math.min(r[2], Math.max(r[0], p.x)), y: Math.min(r[3], Math.max(r[1], p.y)) } : { x: p.x, y: p.y };
    this.grip = grip;
    this.dirty = true;
  }

  /** Once a frame: send what changed, keep the hold alive, draw the marks on the table. */
  frame(now, seconds) {
    if (this.active && this.at && (this.dirty || now - this.last > 400) && now - this.last >= SEND_MS) {
      this.last = now;
      this.dirty = false;
      const value = { x: this.at.x, y: this.at.y, grip: this.grip };
      if (this.own) value.own = true;
      const camera = this.camera;
      if (camera) {
        value.input = 'camera';
        if (camera.world && now - camera.seen < 400) {       // the hand itself, for the robot's fingers
          value.world = camera.world;
          value.right = camera.right;
          value.depth = camera.depth;
        }
      }
      this.send({ type: 'cmd', name: 'hand', value });
    }
    this.marks.update(this.active ? this : null, seconds);
  }

  /** The part a press would take now. */
  nearest() {
    if (!this.at || !this.table) return null;
    let best = null;
    let bestD = NEAR_M;
    for (const part of this.table.parts) {
      if (!part.on_mat || part.held) continue;
      const d = Math.hypot(part.x - this.at.x, part.y - this.at.y);
      if (d < bestD) {
        bestD = d;
        best = part;
      }
    }
    return best;
  }

  bind() {
    const canvas = this.canvas;
    canvas.addEventListener('pointerdown', (event) => {
      if (!this.active || this.camera) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      this.pointer = event.pointerId;
      canvas.setPointerCapture(event.pointerId);
      this.aim(event.clientX, event.clientY, true);
    });
    canvas.addEventListener('pointermove', (event) => {
      if (!this.active || this.camera) return;
      if (this.pointer !== null && this.pointer !== event.pointerId) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      // a mouse that only hovers still moves the hand: it waits over the part before the press
      this.aim(event.clientX, event.clientY, this.pointer === event.pointerId);
    });
    const release = (event) => {
      if (!this.active || this.pointer !== event.pointerId) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      this.aim(event.clientX, event.clientY, false);
      this.pointer = null;
    };
    canvas.addEventListener('pointerup', release);
    canvas.addEventListener('pointercancel', release);
  }

  // ------------------------------------------------------------------ a hand in front of the camera

  /** Drive with a hand the camera sees. The picture never leaves this page. */
  async startCamera(holder) {
    if (this.camera || !this.active) return false;
    if (!navigator.mediaDevices?.getUserMedia) {
      this.onNote('This browser gives the page no camera. The pointer still drives.');
      return false;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
    } catch {
      this.onNote('The camera did not start. The pointer still drives.');
      return false;
    }
    let landmarker;
    try {
      const { FilesetResolver, HandLandmarker } = await import(`${MEDIAPIPE}vision_bundle.mjs`);
      const files = await FilesetResolver.forVisionTasks(`${MEDIAPIPE}wasm`);
      const make = (delegate) => HandLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: `${MEDIAPIPE}hand_landmarker.task`, delegate },
        runningMode: 'VIDEO',
        numHands: 1,
      });
      try {
        landmarker = await make('GPU');
      } catch {
        landmarker = await make('CPU');
      }
    } catch (error) {
      console.error(error);
      for (const track of stream.getTracks()) track.stop();
      this.onNote('Hand tracking did not load. The pointer still drives.');
      return false;
    }
    if (!this.active) {                       // the turn ended while it loaded
      for (const track of stream.getTracks()) track.stop();
      landmarker.close();
      return false;
    }
    const video = holder.querySelector('video');
    const overlay = holder.querySelector('canvas');
    video.srcObject = stream;
    await video.play().catch(() => {});
    // A phone held upright gives a tall picture, a laptop a wide one. The inset takes the picture's own
    // shape, so none of it is cut away and the hand drawn over it is where the hand is.
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      holder.style.setProperty('--cam-aspect', `${video.videoWidth} / ${video.videoHeight}`);
    }
    this.camera = { stream, landmarker, video, overlay, holder, smooth: null, seen: 0, stamp: -1,
      world: null, right: true, depth: 0, scale0: null, scales: [] };
    holder.hidden = false;
    this.pointer = null;
    this.grip = false;
    this.look();
    return true;
  }

  stopCamera() {
    const camera = this.camera;
    if (!camera) return;
    this.camera = null;
    for (const track of camera.stream.getTracks()) track.stop();
    camera.video.srcObject = null;
    camera.holder.hidden = true;
    try {
      camera.landmarker.close();
    } catch {
      // already gone
    }
    this.grip = false;
    // Their own grasp needs their fingers. Without the camera only a glove gives them: back to the helped
    // grasp, which the switch on the page shows, so a pointer is never left waiting for fingers it has not got.
    if (!this.glove) this.own = false;
    this.dirty = true;
  }

  look() {
    const camera = this.camera;
    if (!camera) return;
    requestAnimationFrame(() => this.look());
    const video = camera.video;
    if (video.readyState < 2 || video.currentTime === camera.stamp) return;
    camera.stamp = video.currentTime;
    let hand = null;
    let seen = null;
    try {
      seen = camera.landmarker.detectForVideo(video, performance.now());
      hand = seen.landmarks?.[0] || null;
    } catch {
      return;
    }
    this.draw(camera, hand);
    if (!hand) {
      // A hand that left the picture lets go of nothing: the robot keeps what it has, where it has it, until
      // the hand is seen again. The page says the hand is not seen (hud.js), so nothing here decides for it.
      if (performance.now() - camera.seen > 1500) {           // gone: wherever it comes back is level again
        camera.scale0 = null;
        camera.scales.length = 0;
        camera.world = null;
      }
      return;
    }
    camera.seen = performance.now();
    this.measure(camera, hand, seen, video.videoWidth / Math.max(1, video.videoHeight));
    const reading = readHand(hand, this.grip);
    const s = camera.smooth || { x: reading.x, y: reading.y };
    s.x += (reading.x - s.x) * 0.4;
    s.y += (reading.y - s.y) * 0.4;
    camera.smooth = s;
    // the middle of the picture is the table: nobody has to reach for the picture's edges
    const box = this.reach();
    const u = clamp01((s.x - 0.2) / 0.6);
    const v = clamp01((s.y - 0.25) / 0.55);
    this.aim(box.left + u * box.width, box.top + v * box.height, reading.grip);
  }

  /** The hand itself, for the robot's fingers: its landmarks in metres, which hand it is, how near it is. */
  measure(camera, hand, seen, aspect) {
    const world = seen.worldLandmarks?.[0];
    if (!world || world.length !== 21) {
      camera.world = null;
      return;
    }
    camera.world = world.flatMap((p) => [round4(p.x), round4(p.y), round4(p.z)]);
    // the picture is not mirrored when the model reads it, so its "Left" is the visitor's right hand
    const label = (seen.handedness || seen.handednesses)?.[0]?.[0]?.categoryName;
    camera.right = label !== 'Right';
    // How large the hand is in the picture, from its own shape: twice as large is half as far.
    let ax = 0; let ay = 0; let bx = 0; let by = 0;
    for (let i = 0; i < 21; i++) {
      ax += world[i].x / 21; ay += world[i].y / 21;
      bx += hand[i].x * aspect / 21; by += hand[i].y / 21;
    }
    let dot = 0; let size = 0;
    for (let i = 0; i < 21; i++) {
      const wx = world[i].x - ax; const wy = world[i].y - ay;
      dot += wx * (hand[i].x * aspect - bx) + wy * (hand[i].y - by);
      size += wx * wx + wy * wy;
    }
    const scale = dot / Math.max(size, 1e-9);
    if (!(scale > 0)) return;
    if (camera.scale0 === null) {                               // where the hand is first held is level
      camera.scales.push(scale);
      if (camera.scales.length >= 12) camera.scale0 = camera.scales.sort((a, b) => a - b)[6];
      camera.depth = 0;
      return;
    }
    camera.depth += (Math.log(scale / camera.scale0) - camera.depth) * 0.35;
    this.dirty = true;
  }

  /** Where the hand may go, as a box on the page: the hand in the picture moves over this. */
  reach() {
    const bounds = this.canvas.getBoundingClientRect();
    const r = this.table?.reach || [0.21, -0.35, 0.515, 0.35];
    let left = Infinity; let right = -Infinity; let top = Infinity; let bottom = -Infinity;
    for (const [x, y] of [[r[0], r[1]], [r[0], r[3]], [r[2], r[1]], [r[2], r[3]]]) {
      this.point.set(x, y, this.tableZ).project(this.rig.camera);
      const px = bounds.left + (this.point.x * 0.5 + 0.5) * bounds.width;
      const py = bounds.top + (0.5 - this.point.y * 0.5) * bounds.height;
      left = Math.min(left, px); right = Math.max(right, px); top = Math.min(top, py); bottom = Math.max(bottom, py);
    }
    return { left, top, width: right - left, height: bottom - top };
  }

  draw(camera, hand) {
    const overlay = camera.overlay;
    // the drawing covers the picture exactly: same box, in the screen's own pixels
    const box = camera.video.getBoundingClientRect();
    const w = Math.max(1, Math.round(box.width * devicePixelRatio));
    const h = Math.max(1, Math.round(box.height * devicePixelRatio));
    if (overlay.width !== w || overlay.height !== h) {
      overlay.width = w;
      overlay.height = h;
    }
    const g = overlay.getContext('2d');
    g.clearRect(0, 0, w, h);
    camera.holder.toggleAttribute('data-lost', !hand);
    if (!hand) return;
    const at = pictureToBox(camera.video.videoWidth, camera.video.videoHeight, w, h);
    const pinched = this.grip;
    g.lineWidth = 1.5 * devicePixelRatio;
    g.strokeStyle = pinched ? 'rgba(74, 222, 128, 0.95)' : 'rgba(255, 255, 255, 0.9)';
    g.fillStyle = g.strokeStyle;
    g.beginPath();
    for (const [a, b] of BONES) {
      g.moveTo(...at(hand[a]));
      g.lineTo(...at(hand[b]));
    }
    g.stroke();
    for (const tip of [4, 8]) {
      g.beginPath();
      g.arc(...at(hand[tip]), 3.5 * devicePixelRatio, 0, Math.PI * 2);
      g.fill();
    }
    if (this.own) {
      // how far down the hand is: pushed towards the camera, it fills
      const down = clamp01(camera.depth / 0.30);
      const x = w - 9 * devicePixelRatio; const top = 8 * devicePixelRatio; const tall = h - 16 * devicePixelRatio;
      g.fillStyle = 'rgba(255, 255, 255, 0.28)';
      g.fillRect(x, top, 3 * devicePixelRatio, tall);
      g.fillStyle = 'rgba(255, 255, 255, 0.95)';
      g.fillRect(x, top + tall * (1 - down), 3 * devicePixelRatio, tall * down);
    }
  }
}

/** Where a hand is in the picture (mirrored, 0..1) and whether thumb and index finger pinch. */
export function readHand(hand, wasPinched) {
  const palm = [0, 5, 17].reduce((sum, i) => ({ x: sum.x + hand[i].x / 3, y: sum.y + hand[i].y / 3 }), { x: 0, y: 0 });
  const length = Math.hypot(hand[9].x - hand[0].x, hand[9].y - hand[0].y) || 1e-6;
  const gap = Math.hypot(hand[8].x - hand[4].x, hand[8].y - hand[4].y) / length;
  return { x: 1 - palm.x, y: palm.y, gap, grip: wasPinched ? gap < PINCH_OFF : gap < PINCH_ON };
}

/**
 * Where a point of the camera's picture (0..1 across it and down it) lands in a box that shows the picture
 * mirrored and scaled to cover: the box keeps the picture's shape, so whatever does not fit is cut away
 * evenly on both sides, and a point must move with it.
 */
export function pictureToBox(pictureW, pictureH, boxW, boxH) {
  const scale = pictureW > 0 && pictureH > 0 ? Math.max(boxW / pictureW, boxH / pictureH) : 0;
  const w = scale ? pictureW * scale : boxW;
  const h = scale ? pictureH * scale : boxH;
  const left = (boxW - w) / 2;
  const top = (boxH - h) / 2;
  return (point) => [left + (1 - point.x) * w, top + point.y * h];       // mirrored, like the picture under it
}

const clamp01 = (v) => Math.min(1, Math.max(0, v));
const round4 = (v) => Math.round(v * 10000) / 10000;

// The marks on the driver's table: where the hand is asked to be, the part a press would take,
// the bin the part in the hand belongs in.
class Marks {
  constructor(stage, tableZ) {
    this.z = tableZ + 0.003;
    this.group = new THREE.Group();
    this.group.visible = false;
    const flat = (color, opacity) => new THREE.MeshBasicMaterial({
      color, transparent: true, opacity, depthWrite: false, side: THREE.DoubleSide, toneMapped: false,
    });
    this.cursor = new THREE.Mesh(new THREE.RingGeometry(0.019, 0.023, 48), flat(0xffffff, 0.95));
    this.dot = new THREE.Mesh(new THREE.CircleGeometry(0.008, 24), flat(0xffffff, 0.95));
    this.take = new THREE.Mesh(new THREE.RingGeometry(0.040, 0.045, 64), flat(0xffffff, 0.9));
    this.binLine = new THREE.LineLoop(
      new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(12), 3)),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95, toneMapped: false }),
    );
    this.binFill = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), flat(0xffffff, 0.14));
    for (const mark of [this.cursor, this.dot, this.take, this.binLine, this.binFill]) {
      mark.renderOrder = 20;
      mark.material.depthTest = false;      // the robot's own arm must not hide where the hand is going
      this.group.add(mark);
    }
    stage.scene.add(this.group);
  }

  update(drive, seconds) {
    this.group.visible = Boolean(drive?.at);
    if (!this.group.visible) return;
    const { at, grip, table, want } = drive;
    this.cursor.position.set(at.x, at.y, this.z);
    this.dot.position.set(at.x, at.y, this.z);
    this.dot.visible = grip;
    this.cursor.scale.setScalar(grip ? 0.8 : 1);

    const near = want ? null : drive.nearest();
    this.take.visible = Boolean(near);
    if (near) {
      this.take.position.set(near.x, near.y, this.z);
      this.take.scale.setScalar(1 + 0.06 * Math.sin(seconds * 6));
    }

    const bin = want && table ? table.bins[want] : null;
    this.binLine.visible = this.binFill.visible = Boolean(bin);
    if (bin) {
      const [x0, y0, x1, y1] = bin;
      const over = at.x >= x0 && at.x <= x1 && at.y >= y0 && at.y <= y1;
      const color = over ? 0x4ade80 : 0xffffff;
      const z = this.z + 0.002;
      const p = this.binLine.geometry.attributes.position;
      p.setXYZ(0, x0, y0, z); p.setXYZ(1, x1, y0, z); p.setXYZ(2, x1, y1, z); p.setXYZ(3, x0, y1, z);
      p.needsUpdate = true;
      this.binLine.geometry.computeBoundingSphere();
      this.binLine.material.color.setHex(color);
      this.binFill.material.color.setHex(color);
      this.binFill.material.opacity = over ? 0.28 : 0.10 + 0.06 * (0.5 + 0.5 * Math.sin(seconds * 5));
      this.binFill.position.set((x0 + x1) / 2, (y0 + y1) / 2, z);
      this.binFill.scale.set(x1 - x0, y1 - y0, 1);
    }
  }
}
