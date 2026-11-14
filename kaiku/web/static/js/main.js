// Kaiku's browser client. Loads the scene and its meshes, opens the live stream, and then draws
// a frame at the display's rate: camera, robots, the operator's hands, floor markers, page.

import * as THREE from '../vendor/three/three.module.min.js';
import { CameraRig } from './camera.js';
import { CalibrationCard } from './calibrate.js';
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
const calibration = new CalibrationCard({ app, send: (message) => link.send(message) });

boot().catch((error) => {
  console.error(error);
  hud.loadingFailed('Something went wrong while loading. Reload the page to try again.');
});

async function boot() {
  hud.loading('Loading the scene', 0.06);
  const robots = (n) => `${n} robot${n === 1 ? '' : 's'}`;
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
  const drive = new Drive({
    stage, canvas, rig, send: (message) => link.send(message), tableZ: described.tableZ,
    onNote: (text) => hud.toast(text, 'warn'),
  });
  const stream = new Stream({ nEnvs: described.nEnvs, nq: described.kin.nq, modelHash: described.modelHash });
  link.send = (message) => stream.send(message);
  hud.setStates(stream.state, described.nEnvs, layout);

  hud.loading(`Loading ${robots(described.nEnvs)}`, 0.15);
  const coarse = await untilAnswered(() => fetchPack(2, (bytes) => {
    hud.loading(`Loading ${robots(described.nEnvs)}`, 0.15 + 0.5 * (1 - Math.exp(-bytes / 300000)));
  }));
  hud.loading(`Building ${robots(described.nEnvs)}`, 0.7);
  await sleep(0);                 // let the words reach the screen: building the meshes holds the page for a moment
  fleet.addPack(2, coarse);
  if (flags.preview === 'loading') return;

  // ---------------------------------------------------------------- live data

  let lag = null;
  let running = false;
  let alone = false;              // the policy is driving: no operator, so no echo to draw
  let comparing = null;           // the shot to go back to after the scene without Kaiku and its second half
  stream.onHud = (message) => {
    const data = flags.preview ? previewHud(message, flags.preview, stream.client) : message;
    lag = Array.isArray(data.envs?.lag_s) ? data.envs.lag_s : null;
    running = data.status === 'running';
    if ((data.mode === 'policy') !== alone) {
      // The policy drives every robot, so look at all of them; with an operator, look at the operator.
      alone = data.mode === 'policy';
      if (rig.focus < 0 && rig.shot in rig.shots) rig.goTo(alone ? 'overview' : flags.shot);
    }
    // The scene without Kaiku is about all the robots: look at all of them, then back to where the visitor was.
    const compare = Boolean(data.copy?.on || data.copy?.again);
    if (compare && comparing === null && rig.focus < 0 && rig.shot in rig.shots) {
      comparing = rig.shot;
      if (rig.shot !== 'overview' && rig.shot !== 'top') rig.goTo('overview', 1.2);
    } else if (!compare && comparing !== null) {
      if (rig.focus < 0 && rig.shot === 'overview' && comparing !== 'overview') rig.goTo(comparing, 1.2);
      comparing = null;
    }
    hands.setHands(data.operator?.hands || null, performance.now());
    calibration.update(data);
    // the runner says whose turn it is: this page drives only while it is named
    const driver = data.operator?.driver;
    drive.follow(driver != null && driver === stream.client, data.mode === 'visitor' ? data.operator : null);
    hud.setDrive(drive);
    hud.client = stream.client;
    hud.setHud(data);
  };
  stream.onEvent = (event) => {
    // A word about one page's turn is for that page only, and replaces the last word about it.
    if (event.type === 'toast' && (event.to == null || event.to === stream.client)) hud.toast(event.text, event.level, event.key);
  };
  stream.onStatus = (status) => {
    if (flags.preview !== 'reconnecting' && flags.preview !== 'connecting') hud.setConnection(status);
    if (status === 'live' && flags.reset) {
      // ?reset=1 starts the session over once, then leaves the address: a reload must not do it again
      flags.reset = false;
      stream.send({ type: 'cmd', name: 'reset' });
      const url = new URL(location.href);
      url.searchParams.delete('reset');
      history.replaceState(null, '', url);
    }
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
  const phone = window.matchMedia('(max-width: 760px)');
  hud.onDriveStart = () => drive.start();
  hud.onDriveEnd = () => drive.stop();
  hud.onCamera = () => (drive.camera ? drive.stopCamera() : drive.startCamera(hud.el['hand-cam']));
  drive.onState = (active) => {
    hud.setDriveActive(active);
    // a phone is tall: the table has to fill its width, the robots behind get what is left
    rig.shots.drive.frame = phone.matches ? [1.15, 1.5] : [1.85, 1.6];
    if (active) rig.goTo('drive', 0.9);
    else if (rig.shot === 'drive') rig.goTo(flags.shot, 0.9);
    place();
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
    if (drive.active) {
      if (event.key === 'Escape') drive.stop();
      return;
    }
    if (event.key === '1') rig.goTo('lead');
    else if (event.key === '2') rig.goTo('overview');
    else if (event.key === '3') rig.goTo('top');
    else if (event.key === 'Escape' && rig.focus >= 0) focusOn(-1);
    else if (event.key === 'ArrowRight' && rig.focus >= 0 && onPage) focusOn(rig.focus + 1);
    else if (event.key === 'ArrowLeft' && rig.focus >= 0 && onPage) focusOn(rig.focus - 1);
    else if (event.key === ' ' && onPage) hud.togglePause();
    else if (event.key.toLowerCase() === 'c' && onPage) link.send({ type: 'cmd', name: 'calibrate' });
    else if ((event.key === 'n' || event.key === 'N') && onPage) stream.send({ type: 'cmd', name: 'next_scene' });
    else return;
    event.preventDefault();
  });

  // Where the subject sits in the picture, so the totals laid over the stage do not cover it.
  function place() {
    const close = rig.shot === 'lead' || rig.shot === 'focus';
    if (rig.shot === 'drive') rig.shiftGoal.set(0, phone.matches ? -0.12 : -0.28);   // the table low, the others above it
    else if (phone.matches) rig.shiftGoal.set(0, close ? 0.24 : 0.14);
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
  let drawn = 0;                  // frames on screen with the robots where the runner has them
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
    drive.frame(now, seconds);

    markers.hovered = hovered;
    markers.focused = rig.focus;
    markers.update(stream.state, lag, seconds, dt);

    // the echo rings run while motion is flowing and rest when it is not
    ripple += ((flowing && running && !alone && !still.matches ? 1 : 0) - ripple) * (1 - Math.exp(-dt * 3));
    stage.ring.uRipple.value = ripple;

    stage.aimShadow(rig.target(target), rig.distance);
    stage.render(seconds, rig.distance);
    if (stream.frames > 0) drawn++;

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

  hud.loading('Connecting to the simulation', 0.8);
  stream.connect();
  requestAnimationFrame(frame);
  while (stream.frames === 0) await sleep(50);
  // The first frames compile the shaders and can take seconds: the loading screen stays until the robots
  // are on screen, and until the runner has said who drives (the opening's first line says it).
  hud.loading('Drawing the robots', 0.92);
  const since = performance.now();
  while ((drawn < 3 || !hud.known) && performance.now() - since < 12000) await sleep(50);
  hud.loading('Ready', 1);
  hud.loaded();
  if (opening) {
    const lines = playOpening({
      app,
      line: hud.el.opening,
      rig,
      wide: rig.shots.overview,
      words: () => hud.openingWords(described.nEnvs),
      rest: () => rig.goTo(alone ? 'overview' : flags.shot, 1.3),
    });
    hud.onWords = lines.refresh;
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
