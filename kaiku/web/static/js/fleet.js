// Every robot, its parts and its station, drawn with instanced meshes. One InstancedMesh per robot
// link and level of detail; a robot moves between levels as the camera comes and goes. Nothing
// here allocates per frame.

import * as THREE from '../vendor/three/three.module.min.js';

const LODS = 3;
const FINE_WITHIN = 4.2;        // metres: the nearest robot uses the full meshes inside this
const MEDIUM_WITHIN = 7.5;      // metres: robots inside this use the medium meshes
const TELEPORT = 0.12;          // metres in one frame: a part was placed, it did not move there
const APPEAR_S = 0.42;          // seconds a newly placed part takes to grow in
const WAVE_S = 0.055;           // delay per metre from the operator's robot: a new scene sweeps back

const IDENTITY = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const ORIGIN = new Float64Array(3);

export class Fleet {
  /**
   * @param {import('./stage.js').Stage} stage
   * @param {ReturnType<import('./scene.js').describeScene>} described
   * @param {ReturnType<import('./layout.js').formation>} layout
   */
  constructor(stage, described, layout) {
    this.stage = stage;
    this.kin = described.kin;
    this.described = described;
    this.n = described.nEnvs;
    this.nq = this.kin.nq;
    this.offsets = layout.offsets;
    this.scale = described.quantScale;
    this.groundZ = described.groundZ;
    this.group = new THREE.Group();
    stage.scene.add(this.group);

    this.q = new Float64Array(this.nq);
    this.materials = makeMaterials();
    this.shapes = makeShapes();

    // Level of detail per robot, and one bucket of instanced meshes per level.
    this.lod = new Uint8Array(this.n).fill(2);
    this.distance = new Float32Array(this.n);
    this.packs = new Array(LODS).fill(null);
    this.buckets = new Array(LODS).fill(null);
    this.robotItems = described.robot.map((entry) => ({ entry, body: entry.body, local: null }));

    // Free joints need care when blending two frames: quaternions, and parts that are placed.
    const kin = this.kin;
    this.freeAdr = [];
    this.freeOfBody = new Int32Array(kin.nbody).fill(-1);
    for (let b = 1; b < kin.nbody; b++) {
      if (!kin.isFree(b)) continue;
      this.freeOfBody[b] = this.freeAdr.length;
      this.freeAdr.push(kin.jntQadr[kin.jntAdr[b]]);
    }
    this.appearAt = new Float64Array(this.n * Math.max(1, this.freeAdr.length)).fill(-1e9);
    this.waveDelay = new Float32Array(this.n);
    for (let e = 0; e < this.n; e++) {
      this.waveDelay[e] = WAVE_S * Math.hypot(this.offsets[e * 3], this.offsets[e * 3 + 1]);
    }

    this.partGroups = [];
    this.fixturesBuilt = false;
    this.wristBody = {
      left: findBody(kin, 'left', 'wrist_yaw') ?? findBody(kin, 'left', 'wrist'),
      right: findBody(kin, 'right', 'wrist_yaw') ?? findBody(kin, 'right', 'wrist'),
    };
    /** World position of the operator's robot's wrists, refreshed every frame. */
    this.leadWrist = { left: new THREE.Vector3(0.3, 0.2, 0.15), right: new THREE.Vector3(0.3, -0.2, 0.15) };

    this.seenFrames = -1;
    this.seenBlend = -1;
    this.dirty = true;
    this.animatingUntil = 0;
  }

  /** Add the meshes of one level of detail. The first pack also builds the stations and parts. */
  addPack(lod, geometries) {
    this.packs[lod] = geometries;
    const fine = lod < 2;
    const meshes = this.robotItems.map((item) => {
      const shape = this.shapeOf(item.entry, geometries);
      item.local = localMatrix(item.entry.pos, item.entry.quat, shape.scale);
      const mesh = new THREE.InstancedMesh(shape.geometry, this.materials[item.entry.role], this.n);
      mesh.count = 0;
      mesh.frustumCulled = false;
      mesh.castShadow = true;
      mesh.receiveShadow = fine;
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.group.add(mesh);
      return mesh;
    });
    this.buckets[lod] = { meshes, count: 0 };

    if (!this.fixturesBuilt) {
      this.buildFixtures(geometries);
      this.buildParts(geometries);
      this.fixturesBuilt = true;
    } else if (lod === 1) {
      // parts are small; the medium meshes are plenty and the coarse ones are not round
      for (const group of this.partGroups) {
        if (group.meshId != null && geometries.has(group.meshId)) group.mesh.geometry = geometries.get(group.meshId);
      }
    }
    this.dirty = true;
  }

  shapeOf(entry, geometries) {
    const s = entry.size;
    switch (entry.type) {
      case 'mesh': return { geometry: geometries.get(entry.mesh) || this.shapes.box, scale: [1, 1, 1] };
      case 'box': return { geometry: this.shapes.box, scale: [s[0], s[1], s[2]] };
      case 'sphere': return { geometry: this.shapes.sphere, scale: [s[0], s[0], s[0]] };
      case 'ellipsoid': return { geometry: this.shapes.sphere, scale: [s[0], s[1], s[2]] };
      case 'cylinder': return { geometry: this.shapes.cylinder, scale: [s[0], s[0], s[1]] };
      case 'capsule': return { geometry: this.shapes.capsule(s[0], s[1]), scale: [1, 1, 1] };
      case 'plane': return { geometry: this.shapes.plane, scale: [s[0] || 1, s[1] || 1, 1] };
      case 'part': return this.shapes.part(entry.shape, s[0] * 2);
      default: return { geometry: this.shapes.box, scale: [0.01, 0.01, 0.01] };
    }
  }

  // The furniture of a station never moves: one instance per robot, written once.
  buildFixtures(geometries) {
    const tags = [];
    for (const entry of this.described.fixtures) {
      let shape = this.shapeOf(entry, geometries);
      let local = localMatrix(entry.pos, entry.quat, shape.scale);
      if (entry.role === 'mat') {
        // the printed mat is drawn as its top face only, with the print the right way round
        shape = { geometry: this.shapes.sheet, scale: [entry.size[0], entry.size[1], 1] };
        local = localMatrix([entry.pos[0], entry.pos[1], entry.pos[2] + entry.size[2] + 0.0006], entry.quat, shape.scale);
        this.loadMatPrint();
      }
      const mesh = this.instanced(shape.geometry, this.materials[entry.role], this.n);
      mesh.castShadow = entry.role !== 'mat';
      mesh.receiveShadow = true;
      for (let e = 0; e < this.n; e++) this.writeFixed(mesh, e, local, e);
      mesh.instanceMatrix.needsUpdate = true;

      if (entry.role === 'bin' && entry.binColor) tags.push(entry);
    }
    if (tags.length) this.buildBinTags(tags, geometries);
  }

  // A small plate on the wall of each bin that faces away from the robot, in the colour that
  // belongs in the bin.
  buildBinTags(tags, geometries) {
    const mesh = this.instanced(this.shapes.box, this.materials.tag, tags.length * this.n);
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(tags.length * this.n * 3), 3);
    const color = new THREE.Color();
    const turn = new THREE.Quaternion();
    const box = new THREE.Box3();
    tags.forEach((entry, t) => {
      // the bin's box on the table: its mesh box, turned the way the geom is turned
      const geometry = geometries.get(entry.mesh);
      if (geometry) {
        geometry.computeBoundingBox();
        box.copy(geometry.boundingBox);
      } else {
        box.min.set(-entry.size[0], -entry.size[1], -entry.size[2]);
        box.max.set(entry.size[0], entry.size[1], entry.size[2]);
      }
      turn.set(entry.quat[1], entry.quat[2], entry.quat[3], entry.quat[0]);
      box.applyMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(turn));
      const local = localMatrix(
        [entry.pos[0] + box.max.x + 0.001, entry.pos[1] + (box.min.y + box.max.y) / 2, entry.pos[2] + box.max.z - 0.02],
        [1, 0, 0, 0], [0.001, 0.062, 0.011],
      );
      color.setRGB(entry.binColor[0], entry.binColor[1], entry.binColor[2], THREE.SRGBColorSpace);
      for (let e = 0; e < this.n; e++) {
        const slot = t * this.n + e;
        this.writeFixed(mesh, slot, local, e);
        mesh.instanceColor.setXYZ(slot, color.r, color.g, color.b);
      }
    });
    mesh.instanceMatrix.needsUpdate = true;
  }

  buildParts(geometries) {
    const byShape = new Map();
    for (const entry of this.described.parts) {
      const key = entry.type === 'mesh' ? `mesh${entry.mesh}` : `${entry.type}:${entry.shape}:${entry.size}`;
      if (!byShape.has(key)) byShape.set(key, []);
      byShape.get(key).push(entry);
    }
    const color = new THREE.Color();
    for (const entries of byShape.values()) {
      const shape = this.shapeOf(entries[0], this.packs[1] || geometries);
      const capacity = entries.length * this.n;
      const mesh = this.instanced(shape.geometry, this.materials.part, capacity);
      mesh.count = 0;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
      mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
      const members = entries.map((entry) => {
        color.setRGB(entry.rgba[0], entry.rgba[1], entry.rgba[2], THREE.SRGBColorSpace);
        return {
          body: entry.body,
          free: this.freeOfBody[entry.body],
          local: localMatrix(entry.pos, entry.quat, shape.scale),
          r: color.r, g: color.g, b: color.b,
        };
      });
      this.partGroups.push({ mesh, members, meshId: entries[0].type === 'mesh' ? entries[0].mesh : null });
    }
  }

  instanced(geometry, material, count) {
    const mesh = new THREE.InstancedMesh(geometry, material, count);
    mesh.frustumCulled = false;
    this.group.add(mesh);
    return mesh;
  }

  writeFixed(mesh, slot, local, env) {
    const o = this.offsets;
    writeMatrix(mesh.instanceMatrix.array, slot * 16, IDENTITY, 0, ORIGIN, 0, local, o[env * 3], o[env * 3 + 1], o[env * 3 + 2], 1);
  }

  // The mat's print carries the real bins' orange outlines. On stage the parts own the colour,
  // so the print is drawn in grey.
  loadMatPrint() {
    const texture = this.described.textures[0];
    if (!texture) return;
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext('2d');
      context.drawImage(image, 0, 0);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
      const data = pixels.data;
      for (let i = 0; i < data.length; i += 4) {
        const grey = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
        data[i] = data[i + 1] = data[i + 2] = grey;
      }
      context.putImageData(pixels, 0, 0);
      const map = new THREE.CanvasTexture(canvas);
      map.colorSpace = THREE.SRGBColorSpace;
      map.anisotropy = this.stage.renderer.capabilities.getMaxAnisotropy();
      this.materials.mat.map = map;
      this.materials.mat.color.set(0xb4b4b4);
      this.materials.mat.needsUpdate = true;
    };
    image.src = `/api/tex/${texture.file}`;
  }

  /** Pick each robot's level of detail from its distance to the camera. */
  chooseLods(cameraPosition) {
    const o = this.offsets;
    let nearest = -1;
    let nearestDistance = Infinity;
    for (let e = 0; e < this.n; e++) {
      const d = Math.hypot(cameraPosition.x - o[e * 3] - 0.2, cameraPosition.y - o[e * 3 + 1], cameraPosition.z - o[e * 3 + 2]);
      this.distance[e] = d;
      if (d < nearestDistance) {
        nearestDistance = d;
        nearest = e;
      }
    }
    // a robot keeps its level a little past the threshold, so it does not flicker at the edge
    for (let e = 0; e < this.n; e++) {
      const was = this.lod[e];
      const d = this.distance[e];
      let level = d < MEDIUM_WITHIN * (was <= 1 ? 1.06 : 1) ? 1 : 2;
      if (e === nearest && this.stage.fineMeshes && d < FINE_WITHIN * (was === 0 ? 1.1 : 1)) level = 0;
      if (level !== was) {
        this.lod[e] = level;
        this.dirty = true;
      }
    }
  }

  // The level a robot is drawn at: the one it wants if that pack has arrived, else the next coarser.
  loaded(level) {
    for (let l = level; l < LODS; l++) if (this.buckets[l]) return l;
    for (let l = level - 1; l >= 0; l--) if (this.buckets[l]) return l;
    return -1;
  }

  /** Pose everything from the stream's two newest frames. */
  update(stream, now) {
    if (!this.fixturesBuilt || stream.frames === 0) return;
    const blend = stream.blend(now);
    const fresh = stream.frames !== this.seenFrames;
    const seconds = now / 1000;
    if (!fresh && blend === this.seenBlend && !this.dirty && seconds > this.animatingUntil) return;
    this.seenFrames = stream.frames;
    this.seenBlend = blend;
    this.dirty = false;

    const { kin, nq, q, scale, offsets } = this;
    const { prev, cur } = stream;
    const freeCount = this.freeAdr.length;

    for (let l = 0; l < LODS; l++) if (this.buckets[l]) this.buckets[l].count = 0;
    for (let g = 0; g < this.partGroups.length; g++) this.partGroups[g].mesh.count = 0;

    for (let e = 0; e < this.n; e++) {
      const base = e * nq;
      for (let j = 0; j < nq; j++) {
        const a = prev[base + j];
        q[j] = (a + (cur[base + j] - a) * blend) * scale[j];
      }
      for (let f = 0; f < freeCount; f++) {
        const adr = this.freeAdr[f];
        const a = base + adr;
        const dx = (cur[a] - prev[a]) * scale[adr];
        const dy = (cur[a + 1] - prev[a + 1]) * scale[adr + 1];
        const dz = (cur[a + 2] - prev[a + 2]) * scale[adr + 2];
        if (dx * dx + dy * dy + dz * dz > TELEPORT * TELEPORT) {
          for (let k = 0; k < 7; k++) q[adr + k] = cur[a + k] * scale[adr + k];
          if (fresh) this.appearAt[e * freeCount + f] = seconds + this.waveDelay[e];
          continue;
        }
        // blend quaternions the short way round
        const dot = prev[a + 3] * cur[a + 3] + prev[a + 4] * cur[a + 4] + prev[a + 5] * cur[a + 5] + prev[a + 6] * cur[a + 6];
        if (dot < 0) {
          for (let k = 3; k < 7; k++) q[adr + k] = (-prev[a + k] + (cur[a + k] + prev[a + k]) * blend) * scale[adr + k];
        }
      }
      kin.solve(q);

      const ox = offsets[e * 3], oy = offsets[e * 3 + 1], oz = offsets[e * 3 + 2];
      const level = this.loaded(this.lod[e]);
      if (level >= 0) {
        const bucket = this.buckets[level];
        const slot = bucket.count++ * 16;
        const items = this.robotItems;
        for (let i = 0; i < items.length; i++) {
          const item = items[i];
          writeMatrix(bucket.meshes[i].instanceMatrix.array, slot, kin.xmat, item.body * 9, kin.xpos, item.body * 3, item.local, ox, oy, oz, 1);
        }
      }

      for (let g = 0; g < this.partGroups.length; g++) {
        const { mesh, members } = this.partGroups[g];
        for (let m = 0; m < members.length; m++) {
          const part = members[m];
          if (kin.xpos[part.body * 3 + 2] < this.groundZ - 0.05) continue;     // resting under the floor
          let size = 1;
          if (part.free >= 0) {
            const age = seconds - this.appearAt[e * freeCount + part.free];
            if (age < 0) continue;
            if (age < APPEAR_S) {
              size = easeOutBack(age / APPEAR_S);
              if (seconds + APPEAR_S > this.animatingUntil) this.animatingUntil = seconds + APPEAR_S;
            }
          }
          const slot = mesh.count++;
          writeMatrix(mesh.instanceMatrix.array, slot * 16, kin.xmat, part.body * 9, kin.xpos, part.body * 3, part.local, ox, oy, oz, size);
          const colors = mesh.instanceColor.array;
          colors[slot * 3] = part.r; colors[slot * 3 + 1] = part.g; colors[slot * 3 + 2] = part.b;
        }
      }

      if (e === 0) {
        this.readWrist('left');
        this.readWrist('right');
      }
    }

    // parts still waiting for their turn in the wave keep the frame loop awake
    if (fresh) {
      for (let i = 0; i < this.appearAt.length; i++) {
        if (this.appearAt[i] > seconds && this.appearAt[i] + APPEAR_S > this.animatingUntil) this.animatingUntil = this.appearAt[i] + APPEAR_S;
      }
    }

    for (let l = 0; l < LODS; l++) {
      const bucket = this.buckets[l];
      if (!bucket) continue;
      for (let i = 0; i < bucket.meshes.length; i++) {
        const mesh = bucket.meshes[i];
        mesh.count = bucket.count;
        mesh.visible = bucket.count > 0;
        if (bucket.count > 0) mesh.instanceMatrix.needsUpdate = true;
      }
    }
    for (let g = 0; g < this.partGroups.length; g++) {
      const mesh = this.partGroups[g].mesh;
      mesh.visible = mesh.count > 0;
      mesh.instanceMatrix.needsUpdate = true;
      mesh.instanceColor.needsUpdate = true;
    }
  }

  // The operator's robot stands at the origin, so its body positions are stage positions.
  readWrist(side) {
    const body = this.wristBody[side];
    if (body == null) return;
    const p = this.kin.xpos;
    this.leadWrist[side].set(p[body * 3], p[body * 3 + 1], p[body * 3 + 2]);
  }

  /** The robot a ray hits first, or -1. Each station is tested as one box. */
  pick(origin, direction) {
    const o = this.offsets;
    let best = -1;
    let bestT = Infinity;
    for (let e = 0; e < this.n; e++) {
      const t = rayBox(origin, direction,
        o[e * 3] - 0.34, o[e * 3 + 1] - 0.62, o[e * 3 + 2] + this.groundZ,
        o[e * 3] + 0.76, o[e * 3 + 1] + 0.62, o[e * 3 + 2] + 0.58);
      if (t >= 0 && t < bestT) {
        bestT = t;
        best = e;
      }
    }
    return best;
  }
}

function findBody(kin, side, part) {
  let found = null;
  kin.names.forEach((name, b) => {
    if (name.includes(side) && name.includes(part)) found = b;
  });
  return found;
}

function makeMaterials() {
  const standard = (color, roughness, metalness = 0) => new THREE.MeshStandardMaterial({ color, roughness, metalness });
  return {
    'robot-shell': standard(0xc6c9cf, 0.36, 0.3),
    'robot-dark': standard(0x1c1e22, 0.44, 0.35),
    part: standard(0xffffff, 0.48),
    counter: standard(0x9d9b96, 0.66),
    leg: standard(0x2e3035, 0.42, 0.7),
    bin: standard(0x3a3d43, 0.6, 0.05),
    mat: standard(0x8c8c8c, 0.92),
    tag: standard(0xffffff, 0.5),
  };
}

function makeShapes() {
  const capsules = new Map();
  const sheet = new THREE.PlaneGeometry(2, 2);
  // texture u runs along model y (the mat's long side), v along model x (away from the robot)
  const uv = sheet.attributes.uv;
  const position = sheet.attributes.position;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, (position.getY(i) + 1) / 2, (position.getX(i) + 1) / 2);

  return {
    box: new THREE.BoxGeometry(2, 2, 2),
    sphere: new THREE.SphereGeometry(1, 24, 16),
    cylinder: new THREE.CylinderGeometry(1, 1, 2, 40).rotateX(Math.PI / 2),
    plane: new THREE.PlaneGeometry(2, 2),
    sheet,
    capsule(radius, halfLength) {
      const key = `${radius}:${halfLength}`;
      if (!capsules.has(key)) capsules.set(key, new THREE.CapsuleGeometry(radius, 2 * halfLength, 6, 16).rotateX(Math.PI / 2));
      return capsules.get(key);
    },
    // parts drawn from their body name, for servers that do not export their meshes
    part(shape, size) {
      const h = size / 2;
      if (shape === 'cylinder') return { geometry: this.cylinder, scale: [h, h, h] };
      if (shape === 'triangle') {
        this.prism ??= new THREE.CylinderGeometry(1, 1, 2, 3).rotateX(Math.PI / 2);
        return { geometry: this.prism, scale: [size / 1.5, size / 1.5, h] };
      }
      return { geometry: this.box, scale: [h, h, h] };
    },
  };
}

/** T(pos) * R(quat) * S(scale) as a row-major 3x4. */
function localMatrix(pos, quat, scale) {
  const [w, x, y, z] = quat;
  const m = new Float64Array(12);
  m[0] = (1 - 2 * (y * y + z * z)) * scale[0]; m[1] = 2 * (x * y - w * z) * scale[1]; m[2] = 2 * (x * z + w * y) * scale[2]; m[3] = pos[0];
  m[4] = 2 * (x * y + w * z) * scale[0]; m[5] = (1 - 2 * (x * x + z * z)) * scale[1]; m[6] = 2 * (y * z - w * x) * scale[2]; m[7] = pos[1];
  m[8] = 2 * (x * z - w * y) * scale[0]; m[9] = 2 * (y * z + w * x) * scale[1]; m[10] = (1 - 2 * (x * x + y * y)) * scale[2]; m[11] = pos[2];
  return m;
}

/**
 * out[o..o+15] = station offset * body (R at r, P at p) * local * size, column major as WebGL wants it.
 */
function writeMatrix(out, o, R, r, P, p, L, ox, oy, oz, size) {
  const r0 = R[r], r1 = R[r + 1], r2 = R[r + 2];
  const r3 = R[r + 3], r4 = R[r + 4], r5 = R[r + 5];
  const r6 = R[r + 6], r7 = R[r + 7], r8 = R[r + 8];
  out[o] = (r0 * L[0] + r1 * L[4] + r2 * L[8]) * size;
  out[o + 1] = (r3 * L[0] + r4 * L[4] + r5 * L[8]) * size;
  out[o + 2] = (r6 * L[0] + r7 * L[4] + r8 * L[8]) * size;
  out[o + 3] = 0;
  out[o + 4] = (r0 * L[1] + r1 * L[5] + r2 * L[9]) * size;
  out[o + 5] = (r3 * L[1] + r4 * L[5] + r5 * L[9]) * size;
  out[o + 6] = (r6 * L[1] + r7 * L[5] + r8 * L[9]) * size;
  out[o + 7] = 0;
  out[o + 8] = (r0 * L[2] + r1 * L[6] + r2 * L[10]) * size;
  out[o + 9] = (r3 * L[2] + r4 * L[6] + r5 * L[10]) * size;
  out[o + 10] = (r6 * L[2] + r7 * L[6] + r8 * L[10]) * size;
  out[o + 11] = 0;
  out[o + 12] = (r0 * L[3] + r1 * L[7] + r2 * L[11]) * size + P[p] + ox;
  out[o + 13] = (r3 * L[3] + r4 * L[7] + r5 * L[11]) * size + P[p + 1] + oy;
  out[o + 14] = (r6 * L[3] + r7 * L[7] + r8 * L[11]) * size + P[p + 2] + oz;
  out[o + 15] = 1;
}

function easeOutBack(t) {
  const c = 1.4;
  const u = t - 1;
  return 1 + (c + 1) * u * u * u + c * u * u;
}

// Distance along the ray to an axis-aligned box, or -1.
function rayBox(origin, direction, x0, y0, z0, x1, y1, z1) {
  let near = 0;
  let far = Infinity;
  const lo = [x0, y0, z0];
  const hi = [x1, y1, z1];
  const from = [origin.x, origin.y, origin.z];
  const along = [direction.x, direction.y, direction.z];
  for (let axis = 0; axis < 3; axis++) {
    if (Math.abs(along[axis]) < 1e-9) {
      if (from[axis] < lo[axis] || from[axis] > hi[axis]) return -1;
      continue;
    }
    let a = (lo[axis] - from[axis]) / along[axis];
    let b = (hi[axis] - from[axis]) / along[axis];
    if (a > b) [a, b] = [b, a];
    near = Math.max(near, a);
    far = Math.min(far, b);
    if (near > far) return -1;
  }
  return near;
}
