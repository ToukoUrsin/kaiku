// Loads what the server knows about the model (/api/scene) and the mesh packs, and sorts the
// geoms into what the stage draws: the robot, the parts it sorts, and the furniture of one station.

import * as THREE from '../vendor/three/three.module.min.js';
import { Kinematics } from './fk.js';

const PART_SIZE = 0.045;
const PART_COLORS = {
  red: [200 / 255, 30 / 255, 30 / 255, 1],
  yellow: [235 / 255, 200 / 255, 20 / 255, 1],
  blue: [30 / 255, 70 / 255, 200 / 255, 1],
};
// Which colour each bin takes (the challenge's fixed mapping: red left, yellow centre, blue right).
const BIN_COLORS = { left: 'red', center: 'yellow', right: 'blue' };

export async function fetchScene() {
  const response = await fetch('/api/scene', { cache: 'no-cache' });
  if (!response.ok) throw new Error(`scene: HTTP ${response.status}`);
  return response.json();
}

/**
 * Sort the scene's geoms by the job they have on stage.
 * Returns { kin, robot, parts, fixtures, groundZ, quantScale, nEnvs }.
 * Every entry has { body, type, size, pos, quat, rgba, mesh?, role, name }.
 */
export function describeScene(scene) {
  const kin = new Kinematics(scene.tree);
  const hasChildren = new Uint8Array(kin.nbody);
  for (let b = 1; b < kin.nbody; b++) hasChildren[kin.parent[b]] = 1;
  const isPart = (b) => b > 0 && kin.isFree(b) && !hasChildren[b];

  const robot = [];
  const parts = [];
  const fixtures = [];
  for (const geom of scene.geoms) {
    const entry = { ...geom };
    if (geom.body === 0) {
      if (isStageFloor(geom)) continue;
      entry.role = fixtureRole(geom);
      if (entry.role === 'bin') entry.binColor = PART_COLORS[BIN_COLORS[geom.name.slice(4)]] || null;
      fixtures.push(entry);
    } else if (isPart(geom.body)) {
      entry.role = 'part';
      parts.push(entry);
    } else {
      entry.role = luminance(geom.rgba) < 0.35 ? 'robot-dark' : 'robot-shell';
      robot.push(entry);
    }
  }

  // Servers that do not export the parts (they rest under the floor between episodes) still
  // name their bodies obj_<colour>_<shape>_<n>; draw those from the name.
  if (parts.length === 0) {
    for (let b = 1; b < kin.nbody; b++) {
      const match = /^obj_(red|yellow|blue)_(cube|cylinder|triangle)_\d+$/.exec(kin.names[b]);
      if (!match || !isPart(b)) continue;
      parts.push({
        body: b, role: 'part', type: 'part', shape: match[2], name: kin.names[b],
        size: [PART_SIZE / 2, PART_SIZE / 2, PART_SIZE / 2], pos: [0, 0, 0], quat: [1, 0, 0, 0],
        rgba: PART_COLORS[match[1]],
      });
    }
  }

  return {
    kin, robot, parts, fixtures,
    groundZ: scene.ground_z ?? scene.bounds?.min?.[2] ?? 0,
    tableZ: scene.table_z ?? scene.ground_z ?? scene.bounds?.min?.[2] ?? 0,
    quantScale: Float64Array.from(scene.quant.scale),
    nEnvs: scene.n_envs || 1,
    textures: scene.textures || [],
    modelHash: scene.model_hash,
    title: scene.title || 'Kaiku',
  };
}

function isStageFloor(geom) {
  if (geom.type === 'plane' || geom.name === 'floor') return true;
  return geom.type === 'box' && Math.max(geom.size[0], geom.size[1]) >= 1.5;
}

function fixtureRole(geom) {
  if (geom.name === 'mat' || geom.tex >= 0) return 'mat';
  if (geom.name.startsWith('bin_')) return 'bin';
  if (geom.name === 'counter') return 'counter';
  return luminance(geom.rgba) < 0.35 ? 'leg' : 'counter';
}

function luminance(rgba) {
  return 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2];
}

/**
 * Fetch one mesh pack (see scene_export.py for the layout). Resolves to Map(mesh id -> BufferGeometry).
 * `onProgress(bytes)` is called as the download advances.
 */
export async function fetchPack(lod, onProgress = () => {}) {
  const response = await fetch(`/api/pack/lod${lod}.bin`);
  if (!response.ok) throw new Error(`mesh pack ${lod}: HTTP ${response.status}`);
  let buffer;
  if (response.body) {
    const reader = response.body.getReader();
    const chunks = [];
    let total = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      total += value.length;
      onProgress(total);
    }
    const bytes = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }
    buffer = bytes.buffer;
  } else {
    buffer = await response.arrayBuffer();
  }
  return parsePack(buffer);
}

function parsePack(buffer) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== 'KMSH') throw new Error('not a kaiku mesh pack');
  const jsonLength = view.getUint32(8, true);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 12, jsonLength)));
  const blobs = 12 + jsonLength;

  const geometries = new Map();
  for (const mesh of head.meshes) {
    const quantised = new Int16Array(buffer, blobs + mesh.pos, mesh.nv * 3);
    const positions = new Float32Array(mesh.nv * 3);
    for (let i = 0; i < positions.length; i++) {
      const axis = i % 3;
      positions[i] = mesh.center[axis] + mesh.half[axis] * (quantised[i] / 32767);
    }
    const normals = new Int8Array(buffer, blobs + mesh.nrm, mesh.nv * 3);
    const Index = mesh.idx_bytes === 2 ? Uint16Array : Uint32Array;
    const index = new Index(buffer, blobs + mesh.idx, mesh.nf * 3);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3, true));
    geometry.setIndex(new THREE.BufferAttribute(index, 1));
    if (mesh.uv != null) {
      geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(buffer, blobs + mesh.uv, mesh.nv * 2), 2));
    }
    geometries.set(mesh.id, geometry);
  }
  return geometries;
}
