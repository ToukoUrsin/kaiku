// The studio: renderer, light, floor and the tiers the ranks stand on. Model coordinates are used
// as they are (z up), so nothing is ever rotated between MuJoCo and the screen.

import * as THREE from '../vendor/three/three.module.min.js';
import { FIRST_RANK, RANK_PITCH, TIER_LIP, TIER_RISE } from './layout.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

export const COLORS = {
  background: 0x0c0d10,       // --bg in tokens.css
  floor: 0x4a4d55,
  ok: 0x4ade80,
  fail: 0xf0694a,
};

// The echo: rings that leave the operator's robot and cross the ranks. One definition, used by
// the floor and by the station markers so both light up at the same moment.
export const RING_GLSL = /* glsl */ `
  const float RING_SPEED = 5.5;     // metres per second
  const float RING_PERIOD = 1.9;    // seconds between rings
  float kaikuRing(float dist, float time) {
    float phase = fract((time - dist / RING_SPEED) / RING_PERIOD);
    // metres behind the wave front (negative: the front has not arrived yet)
    float behind = (phase < 0.5 ? phase : phase - 1.0) * RING_PERIOD * RING_SPEED;
    float band = behind >= 0.0 ? exp(-behind * 4.5) : exp(behind * 9.0);
    return band * smoothstep(0.7, 1.4, dist) / (1.0 + 0.09 * dist);
  }
`;

export class Stage {
  constructor(canvas, { groundZ, ranks }) {
    this.canvas = canvas;
    this.groundZ = groundZ;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NeutralToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer = renderer;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(COLORS.background);
    this.scene.fog = new THREE.Fog(COLORS.background, 8, 34);

    this.camera = new THREE.PerspectiveCamera(32, 1, 0.1, 120);
    this.camera.up.set(0, 0, 1);

    // Shared by every shader that draws the echo rings.
    this.ring = { uTime: { value: 0 }, uRipple: { value: 0 } };

    this.pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    this.width = 1;
    this.height = 1;
    this.frameMs = 16;          // smoothed cost of a frame, drives the quality steps
    this.quality = 3;           // 3 full, 2 lower resolution, 1 no fine meshes, 0 no shadows
    this.slowFrames = 0;

    this.buildLights();
    this.buildEnvironment();
    this.buildFloor(ranks);
  }

  buildLights() {
    const scene = this.scene;
    const hemisphere = new THREE.HemisphereLight(0xcfd8ea, 0x15161a, 0.38);
    hemisphere.position.set(0, 0, 1);
    scene.add(hemisphere);

    // Key: high, in front of the robots and to their left, across from where the camera usually
    // stands, so shadows fall towards the viewer. The only light that casts shadows, and its
    // shadow box follows whatever the camera is looking at.
    const key = new THREE.DirectionalLight(0xfff3e4, 3.0);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.bias = -0.0003;
    key.shadow.normalBias = 0.012;
    key.shadow.radius = 2.5;
    const box = key.shadow.camera;
    box.left = -3.6; box.right = 3.6; box.top = 3.6; box.bottom = -3.6;
    box.near = 1; box.far = 22;
    box.up.set(0, 0, 1);
    scene.add(key, key.target);
    this.key = key;
    this.keyOffset = new THREE.Vector3(2.6, 5.2, 7.6);

    const fill = new THREE.DirectionalLight(0xbcd0ff, 0.45);
    fill.position.set(5, -6, 2.5);
    scene.add(fill);

    // Rim: from behind the ranks, so every shell has a bright edge against the dark.
    const rim = new THREE.DirectionalLight(0xdbe6ff, 1.25);
    rim.position.set(-8, -3, 4.5);
    scene.add(rim);

    this.aimShadow(new THREE.Vector3(0.3, 0, 0), 4);
  }

  /**
   * Centre the shadow box on `target` and size it to what the camera sees from `distance`.
   * Both are snapped, so shadows do not crawl as the camera eases.
   */
  aimShadow(target, distance) {
    const step = 0.25;
    const x = Math.round(target.x / step) * step;
    const y = Math.round(target.y / step) * step;
    const z = Math.round(target.z / step) * step;
    this.key.target.position.set(x, y, z);
    this.key.position.set(x + this.keyOffset.x, y + this.keyOffset.y, z + this.keyOffset.z);
    this.key.target.updateMatrixWorld();

    const reach = distance < 7 ? 3.6 : distance < 14 ? 7.5 : 15;
    const box = this.key.shadow.camera;
    if (box.right !== reach) {
      box.left = -reach; box.right = reach; box.top = reach; box.bottom = -reach;
      box.far = 22 + reach;
      box.updateProjectionMatrix();
      this.key.shadow.normalBias = 0.012 * (reach / 3.6);
    }
  }

  // Reflections for the robots' shells: a dark room with a few soft boxes, rendered once.
  buildEnvironment() {
    const room = new THREE.Scene();
    const walls = new THREE.Mesh(
      new THREE.BoxGeometry(40, 40, 24),
      new THREE.MeshBasicMaterial({ color: 0x0b0c0f, side: THREE.BackSide }),
    );
    room.add(walls);
    const softbox = (w, h, x, y, z, strength, tint = 0xffffff) => {
      const panel = new THREE.Mesh(
        new THREE.PlaneGeometry(w, h),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(tint).multiplyScalar(strength), side: THREE.DoubleSide }),
      );
      panel.position.set(x, y, z);
      panel.lookAt(0, 0, 0);
      room.add(panel);
    };
    softbox(14, 8, 3, 5, 11, 7.0, 0xfff4e8);        // overhead key
    softbox(6, 9, 12, -7, 4, 2.2, 0xcfdcff);        // front right fill
    softbox(16, 3, -14, 0, 5, 3.2, 0xdde8ff);       // strip behind the ranks
    softbox(30, 30, 0, 0, -10, 0.22);               // floor bounce

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(room, 0.03).texture;
    this.scene.environmentIntensity = 0.55;
    pmrem.dispose();
    room.traverse((object) => {
      object.geometry?.dispose();
      object.material?.dispose();
    });
  }

  buildFloor(ranks) {
    const material = new THREE.MeshStandardMaterial({ color: COLORS.floor, roughness: 0.78, metalness: 0.0 });
    material.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, this.ring);
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vStagePos;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvStagePos = (modelMatrix * vec4(transformed, 1.0)).xyz;');
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', `#include <common>\nvarying vec3 vStagePos;\nuniform float uTime;\nuniform float uRipple;\n${RING_GLSL}`)
        .replace('#include <color_fragment>', /* glsl */ `#include <color_fragment>
          float stageDist = length(vStagePos.xy);
          // the light pools around the operator's robot and falls away into the room
          diffuseColor.rgb *= mix(1.0, 0.36, smoothstep(1.5, 17.0, stageDist));`)
        .replace('#include <emissivemap_fragment>', /* glsl */ `#include <emissivemap_fragment>
          totalEmissiveRadiance += vec3(0.62, 0.74, 0.92) * (0.1 * uRipple * kaikuRing(stageDist, uTime));`);
    };
    this.floorMaterial = material;

    const ground = new THREE.Mesh(new THREE.PlaneGeometry(160, 160), material);
    ground.position.set(0, 0, this.groundZ);
    ground.receiveShadow = true;
    this.scene.add(ground);

    // One long step per rank. Each reaches back under the next, so only its front edge shows.
    const far = 70;
    ranks.forEach((_, r) => {
      const front = FIRST_RANK - r * RANK_PITCH + TIER_LIP;
      const depth = front + far;
      const tier = new THREE.Mesh(new THREE.BoxGeometry(depth, 160, TIER_RISE), material);
      tier.position.set(front - depth / 2, 0, this.groundZ + (r + 0.5) * TIER_RISE);
      tier.receiveShadow = true;
      this.scene.add(tier);
    });
  }

  resize(width, height) {
    if (width < 2 || height < 2) return;
    this.width = width;
    this.height = height;
    this.renderer.setPixelRatio(this.pixelRatio);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
  }

  /** Step the quality down when frames run long (a laptop GPU with 48 robots must hold 60 fps). */
  adapt(frameMs) {
    this.frameMs += 0.05 * (frameMs - this.frameMs);
    this.slowFrames = this.frameMs > 24 ? this.slowFrames + 1 : 0;
    if (this.slowFrames < 90 || this.quality === 0) return;
    this.slowFrames = 0;
    this.quality -= 1;
    if (this.quality === 2) {
      this.pixelRatio = Math.min(this.pixelRatio, 1.25);
      this.resize(this.width, this.height);
    } else if (this.quality === 0) {
      this.renderer.shadowMap.enabled = false;
      this.key.castShadow = false;
    }
  }

  /** Hold one quality instead of adapting: 'high' for stills and video, 'low' for weak machines. */
  pin(level) {
    if (level !== 'low') return;
    this.quality = 0;
    this.pixelRatio = 1;
    this.renderer.shadowMap.enabled = false;
    this.key.castShadow = false;
  }

  /** May the nearest robot use the full-detail meshes? */
  get fineMeshes() {
    return this.quality >= 2;
  }

  render(seconds, cameraDistance) {
    this.ring.uTime.value = seconds;
    // Depth cueing is measured from what the camera looks at, so an overview is not washed out.
    // so the operator's robot stays crisp and the ranks fall back, and an overview is not washed out.
    const near = Math.max(4.5, cameraDistance * 0.9 + 1.4);
    this.scene.fog.near = near;
    this.scene.fog.far = near + 14 + cameraDistance * 1.2;
    this.renderer.render(this.scene, this.camera);
  }
}
