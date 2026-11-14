// Kaiku's browser client. Loads the scene and its meshes, opens the live stream, and then draws
// a frame at the display's rate: camera, robots, the operator's hands, floor markers, page.

import * as THREE from '../vendor/three/three.module.min.js';
import { CameraRig } from './camera.js';
import { Fleet } from './fleet.js';
import { readFlags } from './flags.js';
import { Hands } from './hands.js';
import { Hud } from './hud.js';
import { formation } from './layout.js';
import { Markers } from './markers.js';
import { playOpening, wantsOpening } from './opening.js';
import { previewHud } from './sample.js';
import { describeScene, fetchPack, fetchScene } from './scene.js';
import { Stage } from './stage.js';
import { Stream } from './stream.js';
import { Drive } from './drive.js';

const flags = readFlags();
const app = document.getElementById('app');
const stageElement = document.getElementById('stage');
const canvas = document.getElementById('view');
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// `send` is filled in once the stream exists; until then commands report "not connected".
const link = { send: () => false };
const hud = new Hud({ app, flags, send: (message) => link.send(message) });

boot().catch((error) => {
  console.error(error);
  hud.loadingFailed('Something went wrong while loading. Reload the page to try again.');
});

async function boot() {
  hud.loading('Loading the scene', 0.06);
  if (flags.preview === 'error') {
    hud.loadingFailed('The runner is not answering. Trying again');
    return;
  }
  const described = describeScene(await untilAnswered(fetchScene));
  const layout = formation(described.nEnvs);

  let stage;
  try {
    stage = new Stage(canvas, { groundZ: described.groundZ, ranks: layout.ranks });
  } catch {
    hud.loadingFailed('This browser cannot draw 3D here. Turn on WebGL or try another browser.');
    return;
  }
  if (flags.quality !== 'auto') stage.pin(flags.quality);

  const fleet = new Fleet(stage, described, layout);
  const markers = new Markers(stage, layout, described.nEnvs, described.groundZ);
  const hands = new Hands(stage);
  const rig = new CameraRig(stage, canvas, layout);
  const drive = new Drive({ canvas, rig, send: (message) => link.send(message), tableZ: described.tableZ });
  const stream = new Stream({ nEnvs: described.nEnvs, nq: described.kin.nq, modelHash: described.modelHash });
  link.send = (message) => stream.send(message);
  hud.setStates(stream.state, described.nEnvs, layout);

  hud.loading('Loading the robots', 0.15);
  const coarse = await untilAnswered(() => fetchPack(2, (bytes) => {
    hud.loading('Loading the robots', 0.15 + 0.5 * (1 - Math.exp(-bytes / 300000)));
  }));
  fleet.addPack(2, coarse);
  if (flags.preview === 'loading') return;

  // ---------------------------------------------------------------- live data

  let lag = null;
  let running = false;
  let alone = false;              // the policy is driving: no operator, so no echo to draw
  stream.onHud = (message) => {
    const data = flags.preview ? previewHud(message, flags.preview) : message;
    lag = Array.isArray(data.envs?.lag_s) ? data.envs.lag_s : null;
    running = data.status === 'running';
    if ((data.mode === 'policy') !== alone) {
      // The policy drives every robot, so look at all of them; with an operator, look at the operator.
      alone = data.mode === 'policy';
      if (rig.focus < 0 && rig.shot in rig.shots) rig.goTo(alone ? 'overview' : flags.shot);
    }
    hands.setHands(data.operator?.hands || null, performance.now());
    hud.setHud(data);
    if (data.mode === 'visitor' && drive.active && rig.shot !== 'drive') rig.goTo('drive', 0.6);
  };
  stream.onEvent = (event) => {
    if (event.type === 'toast') hud.toast(event.text, event.level);
  };
  stream.onStatus = (status) => {
    if (flags.preview !== 'reconnecting' && flags.preview !== 'connecting') hud.setConnection(status);
  };
  stream.onModelChanged = () => reloadOnce();
  hud.onRetry = () => stream.connect();
  hud.setConnection(flags.preview === 'reconnecting' ? 'reconnecting' : 'connecting');

  // ---------------------------------------------------------------- camera, focus, keys

  const namedShot = { last: flags.shot };
  const focusOn = (env) => {
    if (env < 0) rig.goTo(namedShot.last);
    else rig.focusOn(((env % described.nEnvs) + described.nEnvs) % described.nEnvs);
  };
  rig.onChange = () => {
    if (rig.shot in rig.shots) namedShot.last = rig.shot;
    hud.setShot(rig.shot);
    if (rig.focus !== hud.focus) {
      hud.setFocus(rig.focus);
      stream.send({ type: 'focus', env: Math.max(0, rig.focus) });
    }
    place();
  };
  hud.onShot = (name) => rig.goTo(name);
  hud.onFocus = focusOn;
  hud.onDriveStart = () => {
    drive.start();
    rig.goTo('drive', 0.6);
  };
  hud.onDriveEnd = () => drive.stop();
  drive.onState = (active) => {
    hud.setDriveActive(active);
    if (active) {
      rig.goTo('drive', 0.6);
      app.dataset.sheet = 'tall';
    } else if (rig.shot === 'drive') {
      rig.goTo('lead', 0.6);
      delete app.dataset.sheet;
    }
  };

  const origin = new THREE.Vector3();
  const direction = new THREE.Vector3();
  const robotAt = (x, y) => {
    rig.ray(x, y, origin, direction);
    return fleet.pick(origin, direction);
  };
  let hovered = -1;
  rig.onHover = (x, y) => {
    hovered = x < 0 ? -1 : robotAt(x, y);
    canvas.toggleAttribute('data-hover', hovered >= 0);
  };
  rig.onPick = (x, y) => {
    const env = robotAt(x, y);
    if (env >= 0 && env !== rig.focus) focusOn(env);
  };

  window.addEventListener('keydown', (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const onPage = event.target === document.body || event.target === canvas;
    if (event.key === '1') rig.goTo('lead');
    else if (event.key === '2') rig.goTo('overview');
    else if (event.key === '3') rig.goTo('top');
    else if (event.key === 'Escape' && rig.focus >= 0) focusOn(-1);
    else if (event.key === 'ArrowRight' && rig.focus >= 0 && onPage) focusOn(rig.focus + 1);
    else if (event.key === 'ArrowLeft' && rig.focus >= 0 && onPage) focusOn(rig.focus - 1);
    else if (event.key === ' ' && onPage) hud.togglePause();
    else if ((event.key === 'n' || event.key === 'N') && onPage) stream.send({ type: 'cmd', name: 'next_scene' });
    else return;
    event.preventDefault();
  });

  // Where the subject sits in the picture, so the totals laid over the stage do not cover it.
  const phone = window.matchMedia('(max-width: 760px)');
  function place() {
    const close = rig.shot === 'lead' || rig.shot === 'focus';
    if (phone.matches) rig.shiftGoal.set(0, close ? 0.24 : 0.14);
    else rig.shiftGoal.set(close ? 0.15 : 0.05, close ? 0.1 : 0.06);
  }
  const resize = () => {
    stage.resize(stageElement.clientWidth, stageElement.clientHeight);
    rig.refit();
    place();
  };
  new ResizeObserver(resize).observe(stageElement);
  hud.onLayout = resize;
  resize();

  rig.orbit = flags.orbit;
  const opening = flags.env < 0 && wantsOpening(flags);
  if (flags.env >= 0) focusOn(flags.env);
  else if (opening) rig.look(rig.hands);
  else rig.goTo(flags.shot);

  // ---------------------------------------------------------------- frames

  const target = new THREE.Vector3();
  const anchor = new THREE.Vector3();
  const tag = { x: 0, y: 0, visible: false };
  let last = performance.now();
  let ripple = 0;
  const still = window.matchMedia('(prefers-reduced-motion: reduce)');     // no rings for those who asked

  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(0.1, Math.max(0.001, (now - last) / 1000));
    const frameMs = now - last;
    last = now;
    const seconds = now / 1000;
    const flowing = stream.flowing(now);

    rig.update(dt);
    fleet.chooseLods(stage.camera.position);
    fleet.update(stream, now);
    hands.update(fleet.leadWrist, flowing && running, now, dt);

    markers.hovered = hovered;
    markers.focused = rig.focus;
    markers.update(stream.state, lag, seconds, dt);

    // the echo rings run while motion is flowing and rest when it is not
    ripple += ((flowing && running && !alone && !still.matches ? 1 : 0) - ripple) * (1 - Math.exp(-dt * 3));
    stage.ring.uRipple.value = ripple;

    stage.aimShadow(rig.target(target), rig.distance);
    stage.render(seconds, rig.distance);

    // the operator's caption rides above the hands
    anchor.copy(hands.labelAnchor).project(stage.camera);
    tag.x = (anchor.x * 0.5 + 0.5) * stage.width;
    tag.y = (0.5 - anchor.y * 0.5) * stage.height;
    tag.visible = hands.opacity > 0.5 && anchor.z < 1 && rig.distance < 22
      && (rig.focus <= 0) && rig.shot !== 'top'
      && tag.x > 70 && tag.x < stage.width - 70 && tag.y > 40 && tag.y < stage.height - 20
      && !(tag.y < 104 && tag.x < 420 && !flags.clean);     // not on top of the camera buttons
    hud.setStalled(stream.status === 'live' && stream.frames > 0 && !flowing);
    hud.frame(now, dt, tag);

    if (flags.quality === 'auto' && !document.hidden) stage.adapt(frameMs);
  }

  // ---------------------------------------------------------------- go

  window.kaiku = { stage, fleet, markers, hands, rig, stream, hud, drive };     // for the console

  hud.loading('Connecting to the robots', 0.8);
  stream.connect();
  requestAnimationFrame(frame);
  while (stream.frames === 0) await sleep(50);
  hud.loading('Ready', 1);
  hud.loaded();
  if (opening) {
    playOpening({
      app,
      line: hud.el.opening,
      rig,
      wide: rig.shots.overview,
      words: () => hud.openingWords(described.nEnvs),
      rest: () => rig.goTo(alone ? 'overview' : flags.shot, 1.3),
    });
  }
  if (flags.preview === 'toast') {
    hud.toast('Next scene: 4 parts.');
    hud.toast('Left glove is weak. Move closer to the receiver.', 'warn');
  }

  // the finer meshes arrive after the stage is already moving
  for (const lod of [1, 0]) {
    try {
      fleet.addPack(lod, await fetchPack(lod));
    } catch {
      // the coarser meshes stay; nothing to tell the viewer
    }
  }
}

/** Keep asking until the runner answers: it may still be starting. */
async function untilAnswered(request) {
  for (let attempt = 0; ; attempt++) {
    try {
      return await request();
    } catch {
      hud.loadingFailed('The runner is not answering. Trying again');
      await sleep(Math.min(4000, 600 * (attempt + 1)));
    }
  }
}

// The runner was restarted with another model: what is on the page no longer matches it.
function reloadOnce() {
  let recently = 0;
  try {
    recently = Number(sessionStorage.getItem('kaiku-reloaded')) || 0;
    sessionStorage.setItem('kaiku-reloaded', String(Date.now()));
  } catch {
    // private windows may refuse storage; reload anyway
  }
  if (Date.now() - recently > 10000) location.reload();
}
