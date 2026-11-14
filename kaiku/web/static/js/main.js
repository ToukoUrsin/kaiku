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
    if (event.type !== 'toast' || (event.to != null && event.to !== stream.client)) return;
    // Everybody shares the session. The page whose button it was is told what it did; the others are told
    // that another visitor did it, or their robots would stop or change hands for no reason they can see.
    const mine = event.by == null || event.by === stream.client;
    hud.toast(mine ? event.text : event.other || event.text, event.level, event.key);
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
    // A phone is tall: the part of the table the hand reaches fills its width, so a part is a fingertip wide,
    // and the camera looks more steeply down, so the lead robot stands under the words and not behind them.
    rig.shots.drive.frame = phone.matches ? [0.84, 1.2] : [1.85, 1.6];
    rig.shots.drive.polar = (phone.matches ? 50 : 63) * Math.PI / 180;
    wordsEnd = 0;
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
    const typing = Boolean(event.target.closest?.('input, textarea, select, [contenteditable]'));
    if (drive.active) {
      if (event.key === 'Escape') drive.stop();
      return;
    }
    if (event.key === '1') rig.goTo('lead');
    else if (event.key === '2') rig.goTo('overview');
    else if (event.key === '3') rig.goTo('top');
    else if (event.key === 'Escape' && rig.focus >= 0) focusOn(-1);
    // The arrows go from robot to robot wherever the keyboard is, also on a button that was just clicked.
    else if (event.key === 'ArrowRight' && rig.focus >= 0 && !typing) focusOn(rig.focus + 1);
    else if (event.key === 'ArrowLeft' && rig.focus >= 0 && !typing) focusOn(rig.focus - 1);
    else if (event.key === ' ' && onPage) hud.togglePause();
    else if (event.key.toLowerCase() === 'c' && onPage) link.send({ type: 'cmd', name: 'calibrate' });
    else if ((event.key === 'n' || event.key === 'N') && onPage) stream.send({ type: 'cmd', name: 'next_scene' });
    else return;
    event.preventDefault();
  });

  // ---------------------------------------------------------------- the picture and what lies over it

  // A robot's table (top and legs) and a robot with its table, as boxes in its own station's frame.
  const table = { min: [Infinity, Infinity, described.groundZ], max: [-Infinity, -Infinity, described.tableZ] };
  for (const fixture of described.fixtures) {
    if (fixture.role !== 'counter' && fixture.role !== 'leg') continue;
    for (let i = 0; i < 2; i++) {
      table.min[i] = Math.min(table.min[i], fixture.pos[i] - fixture.size[i]);
      table.max[i] = Math.max(table.max[i], fixture.pos[i] + fixture.size[i]);
    }
  }
  if (!Number.isFinite(table.min[0])) Object.assign(table, { min: [0.09, -0.55, described.groundZ], max: [0.69, 0.55, described.tableZ] });
  const HEAD_Z = 0.5;                       // the top of a G1's head over its pelvis
  const station = { min: [-0.22, table.min[1], described.groundZ], max: [table.max[0], table.max[1], HEAD_Z] };
  const corner = new THREE.Vector3();

  /** The box around `box` (at station `env`) on the stage, in pixels, where the camera is going. Not slid. */
  function onStage(box, env) {
    const ghost = rig.arrived();
    const o = layout.offsets;
    const out = { left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity };
    for (let i = 0; i < 8; i++) {
      corner.set(
        (i & 1 ? box.max[0] : box.min[0]) + o[env * 3],
        (i & 2 ? box.max[1] : box.min[1]) + o[env * 3 + 1],
        (i & 4 ? box.max[2] : box.min[2]) + o[env * 3 + 2],
      ).project(ghost);
      const x = (corner.x * 0.5 + 0.5) * stage.width;
      const y = (0.5 - corner.y * 0.5) * stage.height;
      out.left = Math.min(out.left, x); out.right = Math.max(out.right, x);
      out.top = Math.min(out.top, y); out.bottom = Math.max(out.bottom, y);
    }
    return out;
  }

  /**
   * Put `box` in the middle of `free` (stage pixels): nearer or further until it fits, then the picture is
   * slid. A box taller than the room keeps its foot in view: that is where the table is. `foot` stands the
   * box on the bottom of the room instead, and leaves the distance alone.
   */
  function centre(box, env, free, foot = false) {
    let seen = onStage(box, env);
    if (!foot) {
      const fill = Math.max((seen.right - seen.left) / (free.right - free.left), (seen.bottom - seen.top) / (free.bottom - free.top));
      if (fill > 0.05) rig.zoom(Math.min(1.8, Math.max(0.6, fill / 0.94)));
      seen = onStage(box, env);
    }
    const tall = foot || seen.bottom - seen.top > free.bottom - free.top;
    const dx = (free.left + free.right) / 2 - (seen.left + seen.right) / 2;
    const dy = tall ? free.bottom - seen.bottom : (free.top + free.bottom) / 2 - (seen.top + seen.bottom) / 2;
    rig.shiftGoal.set((2 * dx) / stage.width, (-2 * dy) / stage.height);
    return { left: seen.left + dx, right: seen.right + dx, top: seen.top + dy, bottom: seen.bottom + dy };
  }

  const within = (element) => {
    const box = element.getBoundingClientRect();
    const frame = stageElement.getBoundingClientRect();
    return { left: box.left - frame.left, right: box.right - frame.left, top: box.top - frame.top, bottom: box.bottom - frame.top };
  };

  // Where the subject sits in the picture, so nothing laid over the stage covers it.
  function place() {
    const width = stage.width;
    const height = stage.height;
    let room = null;                        // the width left of the table for the totals, where a table is near them
    if (rig.shot === 'drive') {
      // The driver's: the lead robot and the table it sorts on, under the words that say what to do next.
      // The words only ever take more room within a turn: the table does not move under a hand that aims.
      wordsEnd = Math.max(wordsEnd, Math.min(within(hud.el['drive-card']).bottom + 8, height * 0.6));
      const free = { left: 0, right: width, top: wordsEnd, bottom: height - (phone.matches ? 16 : 60) };
      centre({ min: [0.0, -0.42, described.tableZ], max: [0.62, 0.42, HEAD_Z] }, 0, free, true);
    } else if (rig.shot === 'focus' && rig.focus >= 0) {
      // One robot and its table, in the middle of what the totals and the robot's card leave free.
      if (phone.matches) {
        const card = within(hud.el.focus);
        const story = within(hud.el.story);
        // beside its card where the card leaves a column, under it where it does not
        const beside = card.left - 16 >= 150;
        centre(station, rig.focus, beside
          ? { left: 8, right: card.left - 8, top: card.top, bottom: Math.max(story.top - 8, card.top + 120) }
          : { left: 8, right: width - 8, top: Math.min(card.bottom + 8, height * 0.45), bottom: Math.max(story.top - 8, height * 0.6) });
      } else {
        room = FOCUS_ROOM;
        const left = hud.el.story.offsetLeft + room + 24;
        centre(station, rig.focus, { left, right: width - 24, top: 24, bottom: height - 24 });
      }
    } else if (phone.matches) {
      rig.shiftGoal.set(0, rig.shot === 'lead' ? 0.24 : 0.14);
    } else {
      if (rig.shot === 'lead') rig.shiftGoal.set(0.15, 0.1);
      else rig.shiftGoal.set(0.05, 0.06);
      // where the lead robot's table comes down beside the totals, they end where it begins
      const seen = onStage(table, 0);
      const story = within(hud.el.story);
      const left = seen.left + (rig.shiftGoal.x * width) / 2;
      const foot = seen.bottom - (rig.shiftGoal.y * height) / 2;
      if (foot > story.top && left > story.left) room = left - story.left - 24;
    }
    hud.fitStory(room);
  }
  const FOCUS_ROOM = 292;                   // the totals as two rows of two, beside a robot looked at closely
  let wordsEnd = 0;                         // where the driver's words end on the stage, in pixels from its top
  const resize = () => {
    wordsEnd = 0;
    stage.resize(stageElement.clientWidth, stageElement.clientHeight);
    rig.refit();
    place();
  };
  new ResizeObserver(resize).observe(stageElement);
  // the words over a driver's table grow and shrink with what they say: the table keeps clear of them
  new ResizeObserver(() => {
    if (rig.shot === 'drive' && within(hud.el['drive-card']).bottom + 8 > wordsEnd) place();
  }).observe(hud.el['drive-card']);
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
