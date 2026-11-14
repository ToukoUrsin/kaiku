// The operator's hands: the 21 points per glove from the HUD, drawn as a skeleton that hovers over
// the hands of the robot it drives, with a thread from each wrist down to the robot's wrist.
// The skeleton is a caption as much as an object: it is drawn over whatever stands behind it, on
// a soft dark patch, so it reads against forty-seven robots.
//
// Glove points are in metres with the wrist at the origin, +x along the fingers, +y to the left
// and +z out of the back of the hand: the same axes as the stage, so a flat hand lies palm down
// with its fingers pointing the way the robot faces. The points arrive at 10 Hz and are blended.

import * as THREE from '../vendor/three/three.module.min.js';

const POINTS = 21;
const SCALE = 1.5;                // the skeleton is drawn larger than life so it reads from afar
const LIFT = 0.46;                // metres above the robot's wrist
const BONE_RADIUS = 0.0021;
const JOINT_RADIUS = 0.0046;
const THREAD_RADIUS = 0.0011;
const PATCH_SIZE = 0.52;          // metres across the dark patch behind each hand
const SIDES = ['left', 'right'];

// wrist, then thumb, index, middle, ring, little: four points each (three joints and the tip)
const BONES = [];
for (let finger = 0; finger < 5; finger++) {
  const first = 1 + finger * 4;
  BONES.push([0, first], [first, first + 1], [first + 1, first + 2], [first + 2, first + 3]);
}
BONES.push([5, 9], [9, 13], [13, 17]);    // across the knuckles

export class Hands {
  constructor(stage) {
    const overlay = { transparent: true, opacity: 0, fog: false, toneMapped: false, depthTest: false, depthWrite: false };
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff, ...overlay });
    const threadMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, ...overlay, depthTest: true });
    const patchMaterial = new THREE.SpriteMaterial({ map: softDisc(), color: 0x08090b, ...overlay });
    this.material = material;
    this.threadMaterial = threadMaterial;
    this.patchMaterial = patchMaterial;

    const rod = new THREE.CylinderGeometry(1, 1, 1, 10, 1, true).rotateX(Math.PI / 2).translate(0, 0, 0.5);
    this.bones = new THREE.InstancedMesh(rod, material, 2 * BONES.length);
    this.joints = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 14, 10), material, 2 * POINTS);
    this.threads = new THREE.InstancedMesh(rod, threadMaterial, 2);
    this.beads = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 10, 8), material, 2);
    this.group = new THREE.Group();
    this.meshes = [this.bones, this.joints, this.threads, this.beads];
    for (const mesh of this.meshes) {
      mesh.frustumCulled = false;
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.renderOrder = mesh === this.threads ? 8 : 10;
      this.group.add(mesh);
    }
    this.patches = SIDES.map(() => {
      const patch = new THREE.Sprite(patchMaterial);
      patch.scale.setScalar(PATCH_SIZE);
      patch.renderOrder = 9;
      this.group.add(patch);
      return patch;
    });
    this.group.visible = false;
    stage.scene.add(this.group);

    this.from = { left: new Float32Array(POINTS * 3), right: new Float32Array(POINTS * 3) };
    this.to = { left: new Float32Array(POINTS * 3), right: new Float32Array(POINTS * 3) };
    this.world = new Float32Array(POINTS * 3);
    this.anchor = { left: new THREE.Vector3(), right: new THREE.Vector3() };
    this.anchored = false;
    this.present = false;
    this.arrivedAt = 0;
    this.interval = 100;
    this.opacity = 0;
    this.tracking = false;
    /** Where the operator's caption sits: above and between the two hands. */
    this.labelAnchor = new THREE.Vector3();
  }

  /** New glove points from the HUD, or null when no hands are being tracked. */
  setHands(hands, now) {
    const usable = hands && SIDES.every((side) => Array.isArray(hands[side]) && hands[side].length >= POINTS);
    if (!usable) {
      this.present = false;
      return;
    }
    const blend = this.blend(now);
    for (const side of SIDES) {
      const from = this.from[side], to = this.to[side], points = hands[side];
      for (let i = 0; i < POINTS; i++) {
        for (let k = 0; k < 3; k++) {
          const j = i * 3 + k;
          from[j] = this.present ? from[j] + (to[j] - from[j]) * blend : points[i][k];
          to[j] = points[i][k];
        }
      }
    }
    if (this.present) this.interval += 0.2 * (Math.min(250, Math.max(40, now - this.arrivedAt)) - this.interval);
    this.arrivedAt = now;
    this.present = true;
  }

  blend(now) {
    const t = (now - this.arrivedAt) / this.interval;
    return t < 0 ? 0 : t > 1 ? 1 : t;
  }

  /**
   * @param {{left: THREE.Vector3, right: THREE.Vector3}} wrists  the operator's robot's wrists
   * @param {boolean} tracking  whether motion is flowing from the hands right now
   */
  update(wrists, tracking, now, dt) {
    this.opacity += ((this.present ? 1 : 0) - this.opacity) * (1 - Math.exp(-dt * 6));
    this.group.visible = this.opacity > 0.01;
    if (!this.group.visible) return;
    this.material.opacity = this.opacity;
    this.threadMaterial.opacity = 0.45 * this.opacity;
    this.patchMaterial.opacity = 0.62 * this.opacity;

    const blend = this.blend(now);
    const follow = this.anchored ? 1 - Math.exp(-dt * 10) : 1;
    this.anchored = true;
    const seconds = now / 1000;

    for (let h = 0; h < SIDES.length; h++) {
      const side = SIDES[h];
      const wrist = wrists[side];
      const anchor = this.anchor[side];
      anchor.x += (wrist.x - anchor.x) * follow;
      anchor.y += (wrist.y - anchor.y) * follow;
      anchor.z += (wrist.z + LIFT - anchor.z) * follow;

      const from = this.from[side], to = this.to[side], world = this.world;
      let cx = 0, cy = 0, cz = 0;
      for (let i = 0; i < POINTS; i++) {
        const j = i * 3;
        world[j] = anchor.x + SCALE * (from[j] + (to[j] - from[j]) * blend);
        world[j + 1] = anchor.y + SCALE * (from[j + 1] + (to[j + 1] - from[j + 1]) * blend);
        world[j + 2] = anchor.z + SCALE * (from[j + 2] + (to[j + 2] - from[j + 2]) * blend);
        cx += world[j]; cy += world[j + 1]; cz += world[j + 2];
        const tip = i > 0 && i % 4 === 0;
        writeBall(this.joints.instanceMatrix.array, (h * POINTS + i) * 16, world[j], world[j + 1], world[j + 2],
          SCALE * (i === 0 ? JOINT_RADIUS * 1.5 : tip ? JOINT_RADIUS * 1.15 : JOINT_RADIUS));
      }
      this.patches[h].position.set(cx / POINTS, cy / POINTS, cz / POINTS);
      for (let b = 0; b < BONES.length; b++) {
        const a = BONES[b][0] * 3, c = BONES[b][1] * 3;
        writeRod(this.bones.instanceMatrix.array, (h * BONES.length + b) * 16,
          world[a], world[a + 1], world[a + 2], world[c], world[c + 1], world[c + 2], SCALE * BONE_RADIUS);
      }

      // the thread from the hand to the robot's wrist, and a bead that runs down it
      writeRod(this.threads.instanceMatrix.array, h * 16, anchor.x, anchor.y, anchor.z, wrist.x, wrist.y, wrist.z, THREAD_RADIUS);
      const along = tracking ? (seconds * 1.1 + h * 0.5) % 1 : 0;
      const eased = along * along * (3 - 2 * along);
      writeBall(this.beads.instanceMatrix.array, h * 16,
        anchor.x + (wrist.x - anchor.x) * eased, anchor.y + (wrist.y - anchor.y) * eased, anchor.z + (wrist.z - anchor.z) * eased,
        tracking ? 0.0052 * Math.sin(Math.PI * along) : 0);
    }

    for (let i = 0; i < this.meshes.length; i++) this.meshes[i].instanceMatrix.needsUpdate = true;
    this.labelAnchor.set(
      (this.anchor.left.x + this.anchor.right.x) / 2 + 0.06,
      (this.anchor.left.y + this.anchor.right.y) / 2,
      Math.max(this.anchor.left.z, this.anchor.right.z) + 0.1,
    );
  }
}

// A round patch that fades out towards its edge.
function softDisc() {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const context = canvas.getContext('2d');
  const fade = context.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  fade.addColorStop(0, 'rgba(255, 255, 255, 1)');
  fade.addColorStop(0.45, 'rgba(255, 255, 255, 0.7)');
  fade.addColorStop(1, 'rgba(255, 255, 255, 0)');
  context.fillStyle = fade;
  context.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

function writeBall(out, o, x, y, z, radius) {
  out[o] = radius; out[o + 1] = 0; out[o + 2] = 0; out[o + 3] = 0;
  out[o + 4] = 0; out[o + 5] = radius; out[o + 6] = 0; out[o + 7] = 0;
  out[o + 8] = 0; out[o + 9] = 0; out[o + 10] = radius; out[o + 11] = 0;
  out[o + 12] = x; out[o + 13] = y; out[o + 14] = z; out[o + 15] = 1;
}

// A unit rod runs along +z from 0 to 1; stretch it from a to b.
function writeRod(out, o, ax, ay, az, bx, by, bz, radius) {
  let zx = bx - ax, zy = by - ay, zz = bz - az;
  const length = Math.hypot(zx, zy, zz) || 1e-6;
  zx /= length; zy /= length; zz /= length;
  // any direction not parallel to the rod gives a frame around it
  let ux = 0, uy = 0, uz = 1;
  if (Math.abs(zz) > 0.9) { ux = 1; uz = 0; }
  let xx = uy * zz - uz * zy, xy = uz * zx - ux * zz, xz = ux * zy - uy * zx;
  const n = Math.hypot(xx, xy, xz);
  xx /= n; xy /= n; xz /= n;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  out[o] = xx * radius; out[o + 1] = xy * radius; out[o + 2] = xz * radius; out[o + 3] = 0;
  out[o + 4] = yx * radius; out[o + 5] = yy * radius; out[o + 6] = yz * radius; out[o + 7] = 0;
  out[o + 8] = zx * length; out[o + 9] = zy * length; out[o + 10] = zz * length; out[o + 11] = 0;
  out[o + 12] = ax; out[o + 13] = ay; out[o + 14] = az; out[o + 15] = 1;
}
