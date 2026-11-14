// Everything on the page that is not the 3D stage: totals, panels, controls, banners and toasts.
// The runner's HUD message arrives ten times a second; any part of it may be missing, and the
// page must still make sense. Elements are found by their data-ref in index.html.

import { drawChart, drawSpark, percent } from './chart.js';
import { IDLE } from './markers.js';
import { samplePolicy } from './sample.js';

const STATE_NAMES = ['idle', 'running', 'success', 'failed', 'lead'];
const STATE_TEXT = ['Waiting for a scene', 'Working', 'Sorted its scene', 'Did not finish', 'Driven by the operator'];
const STEP_ORDER = ['approach', 'descend', 'grasp', 'lift', 'carry', 'release'];
const STEP_TEXT = {
  approach: 'Reaching for the next part',
  descend: 'Lowering onto it',
  grasp: 'Closing the hand',
  lift: 'Lifting it',
  carry: 'Carrying it to its bin',
  release: 'Letting go',
  home: 'Returning to rest',
  idle: 'Waiting for the next scene',
  calibrating: 'Calibrating the gloves',
};
// What moves the operator's robot, in the page's words. The runner names it (operator.drive) from what
// is driving right now; nothing here may claim gloves that are not the source.
//   name: starts a sentence ("... drives one robot."), with its verb
//   panel: the Operator block's line; mine: the same on the page of the visitor who is driving
//   tag: the caption over the hands; gloves: 'real' or 'emulated' when glove rows and calibration apply
const DRIVES = {
  'stand-in': { name: 'A scripted hand in emulated Smartgloves II', tag: 'Scripted hand, emulated gloves', gloves: 'emulated' },
  emulated: { name: 'Emulated Smartgloves II', verb: 'drive', tag: 'Emulated gloves', gloves: 'emulated' },
  gloves: { name: 'One operator in Smartgloves II', panel: 'Smartgloves II, live', tag: 'Operator in Smartgloves II', gloves: 'real' },
  recorded: { name: 'A recorded session', panel: 'A recorded session, replayed', tag: 'Recorded session' },
  'emulated-recorded': { name: 'A recorded emulated glove session', panel: 'A recorded emulated glove session, replayed', tag: 'Recorded emulated session' },
  pointer: { name: "A visitor's pointer", panel: 'A visitor, with a pointer', mine: 'You, with the pointer', tag: 'A visitor, with a pointer' },
  camera: { name: "A visitor's hand, seen by a camera,", panel: "A visitor's hand, seen by a camera", mine: 'Your hand, seen by the camera', tag: "A visitor's hand, by camera" },
  'glove-pointer': { name: 'A visitor with one Smartglove and a pointer', mine: 'You, with one Smartglove and the pointer', tag: 'One Smartglove and a pointer' },
  'glove-camera': { name: 'A visitor with one Smartglove and a camera', mine: 'You, with one Smartglove and the camera', tag: 'One Smartglove and a camera' },
  'glove-coil': { name: 'A visitor with one Smartglove on a Coil Pro', mine: 'You, with one Smartglove on a Coil Pro', tag: 'One Smartglove on a Coil Pro' },
  autopilot: { name: 'Autopilot', panel: 'Autopilot, scripted motion', tag: 'Autopilot' },
  policy: { name: 'A trained policy', panel: 'A trained policy. Nobody is driving', tag: 'Policy' },
};
const VISITORS = ['pointer', 'camera', 'glove-pointer', 'glove-camera', 'glove-coil'];
const STAGE_TEXT = { loading: 'Reading the demonstrations', training: 'Training', testing: 'Testing on scenes it has not seen', done: 'Finishing' };
const TOAST_MS = 4200;

/** Seconds as something a person waits for: "40 s", "3 min". */
function left(seconds) {
  return seconds < 90 ? `${Math.max(5, Math.round(seconds / 5) * 5)} s` : `${Math.round(seconds / 60)} min`;
}

/** What a training run takes, as it is promised before anybody presses Train: "about 25 minutes on this machine". */
function takes(policy) {
  if (!Number.isFinite(policy?.about_min) || policy.about_min < 1.5) return '';
  return `about ${Math.round(policy.about_min)} minutes ${where(policy)}`;
}

function where(policy) {
  return policy?.rented ? 'on rented GPUs' : 'on this machine';
}
const MAX_TOASTS = 3;

export class Hud {
  /**
   * @param {object} opts
   * @param {HTMLElement} opts.app
   * @param {ReturnType<import('./flags.js').readFlags>} opts.flags
   * @param {(message: object) => boolean} opts.send  sends a message to the runner
   */
  constructor({ app, flags, send }) {
    this.app = app;
    this.flags = flags;
    this.send = send;
    this.el = {};
    for (const node of app.querySelectorAll('[data-ref]')) this.el[node.dataset.ref] = node;
    this.written = new Map();       // what each element shows now, so nothing is written twice

    this.hud = {};
    this.states = null;             // Uint8Array from the stream: state of every robot
    this.connection = 'connecting';
    this.stalled = false;
    this.focus = -1;
    this.minimap = [];
    this.historyKey = '';
    this.chartKey = '';
    this.partsChoice = 'auto';
    this.samplePolicy = flags.sample ? samplePolicy() : null;

    this.demos = { shown: 0, goal: 0, known: false, alone: false, tickUntil: 0, gainUntil: 0 };
    this.tag = { x: -1, y: -1, visible: false };

    this.onShot = () => {};
    this.onFocus = () => {};
    this.onRetry = () => {};
    this.onLayout = () => {};
    this.bannerRun = null;
    this.driveLocal = false;
    this.driving = null;            // the Drive, for what the driver is told
    this.onDriveStart = () => {};
    this.onDriveEnd = () => {};
    this.onCamera = () => {};
    this.onWords = () => {};        // what is driving changed: the opening says it as it is now
    this.known = false;             // the runner has said who drives
    this.phone = window.matchMedia('(max-width: 760px)');

    if (flags.clean) app.dataset.clean = '';
    if (flags.sheet) app.dataset.sheet = flags.sheet;
    this.bind();
    this.selectTab(flags.tab);
    this.renderPolicy({});
  }

  // ------------------------------------------------------------------ wiring

  bind() {
    const { el } = this;
    const command = (name, value) => {
      const message = value === undefined ? { type: 'cmd', name } : { type: 'cmd', name, value };
      if (!this.send(message)) this.toast('Not connected. Try again in a moment.', 'warn');
    };

    el.mode.addEventListener('click', (event) => {
      const button = event.target.closest('[data-mode]');
      if (!button || button.getAttribute('aria-checked') === 'true') return;
      button.dataset.pending = '';
      setTimeout(() => delete button.dataset.pending, 4000);      // the runner may say no (no policy yet)
      command('mode', button.dataset.mode);
    });
    el.drive.addEventListener('click', () => command('mode', this.hud.mode === 'policy' ? 'gloves' : 'policy'));
    el.shots.addEventListener('click', (event) => {
      const button = event.target.closest('[data-shot]');
      if (button) this.onShot(button.dataset.shot);
    });
    el.calibrate.addEventListener('click', () => command('calibrate'));
    el.pause.addEventListener('click', () => this.togglePause());
    el.next.addEventListener('click', () => command('next_scene'));
    el.parts.addEventListener('click', (event) => {
      const button = event.target.closest('[data-parts]');
      if (!button) return;
      this.partsChoice = button.dataset.parts;
      for (const other of el.parts.children) other.setAttribute('aria-checked', String(other === button));
      command('parts', this.partsChoice === 'auto' ? 'auto' : Number(this.partsChoice));
    });
    el.record.addEventListener('click', () => command('record', el.record.getAttribute('aria-checked') !== 'true'));
    el.train.addEventListener('click', () => command('train'));
    el['ds-copy'].addEventListener('click', () => this.copyPath());
    el['banner-action'].addEventListener('click', () => this.bannerRun?.());
    el['copy-stage']?.addEventListener('click', () => command('copy_scene'));
    el['drive-stage']?.addEventListener('click', () => this.onDriveStart());
    el['drive-panel']?.addEventListener('click', () => this.onDriveStart());
    el['drive-back']?.addEventListener('click', () => this.onDriveEnd());
    el['drive-panel-back']?.addEventListener('click', () => this.onDriveEnd());
    el['drive-camera']?.addEventListener('click', () => this.onCamera());
    el['drive-own']?.addEventListener('click', () => this.onOwn());

    el['focus-close'].addEventListener('click', () => this.onFocus(-1));
    el['focus-prev'].addEventListener('click', () => this.onFocus(this.focus - 1));
    el['focus-next'].addEventListener('click', () => this.onFocus(this.focus + 1));

    el.tabs.addEventListener('click', (event) => {
      const tab = event.target.closest('[data-tab]');
      if (tab) this.selectTab(tab.dataset.tab);
    });
    el.headline.addEventListener('click', () => this.selectTab('policy'));
    el.tabs.addEventListener('keydown', (event) => {
      if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
      const tabs = [...el.tabs.children];
      const next = tabs[(tabs.indexOf(document.activeElement) + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
      this.selectTab(next.dataset.tab);
      next.focus();
    });
    el['sheet-handle'].addEventListener('click', () => {
      if (this.app.dataset.sheet === 'tall') delete this.app.dataset.sheet;
      else this.app.dataset.sheet = 'tall';
      this.onLayout();
    });
  }

  togglePause() {
    const paused = this.hud.status === 'paused';
    if (!this.send({ type: 'cmd', name: 'pause', value: !paused })) this.toast('Not connected. Try again in a moment.', 'warn');
  }

  selectTab(name) {
    this.app.dataset.tab = name;
    for (const tab of this.el.tabs.children) {
      const selected = tab.dataset.tab === name;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
    }
    for (const pane of this.app.querySelectorAll('[data-pane]')) pane.hidden = pane.dataset.pane !== name;
  }

  async copyPath() {
    const path = this.hud.dataset?.path;
    if (!path) return;
    try {
      await navigator.clipboard.writeText(path);
      this.toast('Path copied.');
    } catch {
      // no clipboard on plain http: select the text so it can be copied by hand
      getSelection()?.selectAllChildren(this.el['ds-path']);
    }
  }

  // ------------------------------------------------------------------ writing to the page

  text(ref, value) {
    if (this.written.get(ref) === value) return;
    this.written.set(ref, value);
    this.el[ref].textContent = value;
  }

  attr(ref, name, value) {
    const key = `${ref}@${name}`;
    if (this.written.get(key) === value) return;
    this.written.set(key, value);
    if (value == null || value === false) this.el[ref].removeAttribute(name);
    else this.el[ref].setAttribute(name, value === true ? '' : value);
  }

  show(ref, visible) {
    const key = `${ref}@hidden`;
    if (this.written.get(key) === visible) return;
    this.written.set(key, visible);
    this.el[ref].hidden = !visible;
  }

  // ------------------------------------------------------------------ live data

  /** The stream's per-robot states; read whenever the HUD is drawn. */
  setStates(states, nEnvs, layout) {
    this.states = states;
    this.nEnvs = nEnvs;
    this.buildMinimap(layout);
  }

  setConnection(status) {
    this.connection = status;
    this.app.dataset.conn = status;
    this.text('conn-text', status === 'live' ? 'Live' : status === 'reconnecting' ? 'Reconnecting' : 'Connecting');
    this.renderBanner();
  }

  setStalled(stalled) {
    if (stalled === this.stalled) return;
    this.stalled = stalled;
    this.renderBanner();
  }

  setHud(hud) {
    this.hud = hud || {};
    this.known = true;
    if (!('ready' in this.app.dataset)) this.app.dataset.ready = '';
    this.renderTopbar();
    this.renderStory();
    this.renderOperator();
    this.renderEpisode();
    this.renderHistory();
    this.renderFleet();
    this.renderDataset();
    this.renderPolicy(this.hud.policy || {});
    this.renderControls();
    this.renderBanner();
    this.renderFocus();
    this.renderResult();
    this.renderCompare();
    this.onWords();
  }

  /** What moves the operator's robot now: one of DRIVES' names, or null before the runner has said. */
  drive() {
    const operator = this.hud.operator || {};
    if (operator.drive) return operator.drive;
    if (this.hud.mode === 'policy') return 'policy';
    if (operator.source === 'visitor') {                      // a runner that does not say it in one word
      const position = operator.input === 'coil' ? 'coil' : operator.input === 'camera' ? 'camera' : 'pointer';
      if (operator.fingers === 'glove') return `glove-${position}`;
      return operator.fingers === 'camera' || position === 'camera' ? 'camera' : 'pointer';
    }
    return operator.source || null;
  }

  /** Why this page's buttons do nothing right now, or null: the runner holds them, so they look held. */
  heldBy() {
    const operator = this.hud.operator || {};
    if (this.hud.copy?.busy) return { why: 'comparison', text: 'Held until the comparison ends, in about a minute.' };
    if (operator.driver != null && !this.driveLocal) {
      const turn = Number.isFinite(operator.turn_left_s) ? `, for ${clock(operator.turn_left_s)} more at most` : '';
      return { why: 'visitor', text: `Held while a visitor drives${turn}.` };
    }
    return null;
  }

  setDrive(drive) {
    this.driving = drive;
  }

  /** The comparison: its result stays in the panel, and is said once over the scene when both halves are in. */
  renderCompare() {
    const last = this.hud.copy?.last;
    this.show('compare', Boolean(last));
    if (last) {
      const robots = last.robots || 0;
      const both = last.echo != null;
      const share = (n) => `scaleX(${robots ? Math.min(1, Math.max(0, n / robots)) : 0})`;
      this.text('compare-copy', `${last.copy} of ${robots}`);
      this.text('compare-echo', both ? `${last.echo} of ${robots}` : 'Running now');
      this.el['compare-copy-bar'].style.transform = share(last.copy);
      this.el['compare-echo-bar'].style.transform = share(both ? last.echo : 0);
      this.text('compare-when', !both ? '' : last.age_s < 90 ? 'Just now' : `${Math.round(last.age_s / 60)} min ago`);
      this.text('compare-note', `Tables sorted, of ${robots}. Without Kaiku every robot gets the operator's joint commands as they are.`);
    }
    if (!last || last.echo == null || last.age_s > 12 || this.compareSaid === last.at || 'opening' in this.app.dataset) return;
    this.compareSaid = last.at;
    const line = this.el.opening;
    line.textContent = `Without Kaiku: ${last.copy} of ${last.robots} tables sorted.\nWith Kaiku: ${last.echo} of ${last.robots}.`;
    line.setAttribute('data-show', '');
    this.app.dataset.saying = '';
    clearTimeout(this.resultTimer);
    this.resultTimer = setTimeout(() => {
      line.removeAttribute('data-show');
      delete this.app.dataset.saying;
    }, 7000);
  }

  /** A Train run that just ended is said once over the scene, where everybody looks. */
  renderResult() {
    const trained = this.hud.policy?.trained;
    const news = trained && (trained.age_s < 20 || this.flags.result);
    if (!news || this.resultSaid === trained.at || 'opening' in this.app.dataset) return;
    this.resultSaid = trained.at;
    const line = this.el.opening;
    line.textContent = `Trained here on ${trained.humans || 0} demonstrations.\n`
      + `${cap(percent(trained.plain))} alone, ${percent(trained.kaiku)} with echoes, on ${trained.scenes || 0} test scenes.`;
    line.setAttribute('data-show', '');
    this.app.dataset.saying = '';                                     // the tools and the banner make room
    clearTimeout(this.resultTimer);
    this.resultTimer = setTimeout(() => {
      line.removeAttribute('data-show');
      delete this.app.dataset.saying;
    }, 9000);
  }

  renderTopbar() {
    const { mode, perf } = this.hud;
    const held = Boolean(this.heldBy());
    for (const button of this.el.mode.children) {
      if (button.disabled !== held) button.disabled = held;
      const checked = button.dataset.mode === (mode === 'visitor' ? 'gloves' : mode);     // a visitor is an operator too
      // Autopilot is a fallback for a glove session that lost its gloves, not a choice to offer.
      if (button.dataset.mode === 'autopilot') button.hidden = !checked;
      if (checked) delete button.dataset.pending;
      if (button.getAttribute('aria-checked') !== String(checked)) button.setAttribute('aria-checked', String(checked));
    }
    this.text('perf', speed(perf, 'The simulation runs'));
  }

  renderStory() {
    const { totals = {}, echo = {}, dataset = {} } = this.hud;
    const robots = echo.robots ?? Math.max(0, (this.nEnvs || 1) - 1);
    this.text('robots', String(robots));

    // With the policy driving there is no operator: the page counts what the robots do alone.
    const alone = this.hud.mode === 'policy';
    const policy = this.hud.policy || {};
    const tally = policy.alone || {};
    this.text('chain-count', alone ? '0' : '1');
    this.text('chain-who', alone ? 'operators' : 'operator');

    // While recording, the count is what is on disk: it carries on where the last session stopped.
    const session = (totals.human_demos || 0) + (totals.robot_demos || 0);
    const demos = alone ? tally.ok || 0 : dataset.recording && dataset.episodes > session ? dataset.episodes : session;
    const counter = this.demos;
    if (counter.alone !== alone) {
      counter.alone = alone;
      counter.known = false;          // another count altogether: show it, do not roll to it
    }
    if (!counter.known) {
      counter.shown = demos;
      counter.known = this.hud.totals != null;
    } else if (demos > counter.goal) {
      const now = performance.now();
      counter.tickUntil = now + 900;
      if (!alone) {                   // "30 +47 scenes sorted" reads as two numbers; a saved-count gain does not
        counter.gainUntil = now + 2600;
        this.text('gain', `+${demos - counter.goal}`);
      }
    }
    counter.goal = demos;
    this.text('demos-label', alone ? 'new scenes sorted with nobody driving'
      : dataset.recording ? 'demonstrations saved' : 'demonstrations');

    const labels = alone ? ['Scenes sorted alone', 'Learned from the operator', 'Learned from echoes']
      : ['Saved per scene', 'Echoes that succeed', 'Saved per minute'];
    const short = alone ? ['Sorted alone', 'From operator', 'From echoes']               // a phone has a quarter of the room
      : ['Per scene', 'Echoes right', 'Per minute'];
    labels.forEach((label, i) => {
      this.text(`stat${i + 1}-label`, label);
      this.text(`stat${i + 1}-label-2`, short[i]);
    });
    const share = tally.scenes ? tally.ok / tally.scenes : policy.success;
    const amplification = alone ? (Number.isFinite(share) ? percent(share) : '0%')
      : Number.isFinite(totals.amplification) ? trim(totals.amplification) : '0';
    const yieldText = alone ? count(policy.humans || 0)
      : Number.isFinite(totals.yield) ? `${Math.round(totals.yield * 100)}%` : '0%';
    const rate = alone ? count(Math.max(0, (policy.demos || 0) - (policy.humans || 0)))
      : Number.isFinite(totals.demos_per_min) ? String(Math.round(totals.demos_per_min)) : '0';
    const clippedSteps = alone ? tally.clipped || 0 : totals.clipped_steps || 0;
    const clipped = count(clippedSteps);
    // Demonstrations saved for each scene the operator did: its own run and one from every robot that kept up.
    const most = !alone && Number.isFinite(totals.amplification) && totals.episodes ? ` of ${robots + 1}` : '';
    this.text('amplification', amplification);
    this.text('amplification-2', amplification);
    this.text('amplification-of', most);
    this.text('amplification-2-of', most);
    this.text('yield', yieldText);
    this.text('yield-2', yieldText);
    this.text('rate', rate);
    this.text('rate-2', rate);
    this.text('clipped', clipped);
    this.text('clipped-2', clipped);
    this.attr('clipped', 'data-alert', clippedSteps > 0);
    this.attr('clipped-2', 'data-alert', clippedSteps > 0);

    const drive = DRIVES[this.drive()] || DRIVES['stand-in'];
    const who = `${drive.name} ${drive.verb || 'drives'} one robot.`;
    // ?clean=1 keeps the film's caption as it was; the page adds where the runs go
    this.text('caption', alone
      ? `Nobody is driving. A policy trained on the demonstrations sorts ${robots} scenes it has never seen.`
      : this.hud.copy?.on
        ? `${who} The other ${robots} get the same joint commands, but their parts lie somewhere else.`
      : `${who} ${robots} robots repeat every motion, each in its own scene.`
        + (this.flags.clean ? '' : ' The clean runs train a policy.'));
  }

  renderOperator() {
    const operator = this.hud.operator || {};
    const name = this.drive();
    const drive = DRIVES[name];
    const mine = this.driveLocal && VISITORS.includes(name);
    const real = drive?.gloves === 'real';
    const linked = Boolean(operator.left?.connected && operator.right?.connected);
    const date = Number.isFinite(operator.date)
      ? new Date(operator.date * 1000).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' }) : '';
    this.text('source-text', !drive ? 'Waiting for the operator'
      : real && !linked ? 'Smartgloves II, not connected'
        : mine && drive.mine ? drive.mine : drive.panel || drive.name);
    this.attr('source', 'data-live', (real && linked) || VISITORS.includes(name));
    this.attr('source', 'data-standin', name === 'stand-in' || name === 'autopilot' || name === 'emulated');
    this.text('tag-text', drive?.tag || 'Operator');

    // Rows for what is driving, and nothing else: glove rows, rates and calibration only when gloves are the source.
    const rows = [];
    if (drive?.gloves) {
      for (const side of ['left', 'right']) {
        const glove = operator[side] || {};
        rows.push({
          label: `${sentence(side)} glove${real ? '' : ', emulated'}`,
          dot: !glove.connected ? 'none' : glove.status === 'tracking' ? 'tracking' : 'weak',
          value: glove.connected ? sentence(glove.status || 'connected') : 'No signal',
          more: glove.connected && Number.isFinite(glove.hz) ? `${Math.round(glove.hz)} Hz` : '',
        });
      }
      rows.push({ label: 'Calibration', value: operator.calibrated ? 'Calibrated' : 'Not calibrated' });
    } else if (name === 'recorded' || name === 'emulated-recorded') {
      if (date) rows.push({ label: 'Recorded', value: date });
    } else if (VISITORS.includes(name)) {
      const glove = operator.glove || {};
      if (name.startsWith('glove-')) {
        rows.push({
          label: 'Fingers', dot: glove.connected === false ? 'none' : 'tracking',
          value: `One ${glove.side ? `${glove.side} ` : ''}Smartglove`,
          more: Number.isFinite(glove.hz) && glove.hz > 0 ? `${Math.round(glove.hz)} Hz` : '',
        });
      } else if (operator.fingers === 'camera') {
        rows.push({ label: 'Fingers', value: 'Camera' });
      }
      rows.push({ label: 'Position', value: name === 'glove-coil' ? 'Coil Pro' : name.endsWith('camera') ? 'Camera' : 'Pointer' });
      rows.push({ label: 'Grasp', value: operator.own ? (mine ? 'Your own fingers' : 'Their own fingers') : 'Assisted' });
      if (Number.isFinite(operator.turn_left_s)) rows.push({ label: 'Turn', value: `${clock(operator.turn_left_s)} left` });
    }
    const key = JSON.stringify(rows);
    if (this.written.get('operator-rows') !== key) {
      this.written.set('operator-rows', key);
      this.el['operator-rows'].replaceChildren(...rows.map((row) => {
        const item = document.createElement('div');
        item.className = 'row';
        const label = document.createElement('dt');
        label.textContent = row.label;
        const value = document.createElement('dd');
        if (row.dot) {
          const dot = document.createElement('span');
          dot.className = 'state-dot';
          if (row.dot !== 'none') dot.dataset.state = row.dot;
          value.append(dot);
        }
        const text = document.createElement('span');
        text.textContent = row.value;
        value.append(text);
        if (row.more) {
          const more = document.createElement('span');
          more.className = 'num';
          more.textContent = row.more;
          value.append(more);
        }
        item.append(label, value);
        return item;
      }));
    }

    // A turn that was asked for starts with the next scene: until then the line above stays what drives now.
    const starting = operator.starting ? (this.driveLocal || this.driving?.waiting
      ? 'Your turn starts with the next scene.' : 'A visitor takes over with the next scene.') : '';
    const trouble = !drive?.gloves ? '' : operator.message || (operator.coil_present === false
      ? 'No Coil Pro. Hands turn but will not move the robot.'
      : operator.stream_stopped ? 'The glove stream stopped. Waiting for it to come back.'
      : operator.missing?.length ? `${operator.missing[0]} glove missing.`
      : operator.rate_low?.length ? 'The glove rate is too low.' : '');
    this.text('operator-message', starting || trouble);
    this.show('operator-message', Boolean(starting || trouble));

    // Calibration is for gloves somebody wears: live or emulated ones, or the one glove of the visitor driving.
    const ownGlove = mine && name.startsWith('glove-');
    const calibrating = this.hud.episode?.step === 'calibrating';
    this.show('calibrate', name === 'gloves' || name === 'emulated' || ownGlove);
    this.text('calibrate', ownGlove ? 'Recentre' : calibrating ? 'Calibrating' : operator.calibrated ? 'Recalibrate' : 'Calibrate');
    this.attr('calibrate', 'disabled', calibrating);
    this.renderDrive();
  }

  /** The visitor's own turn at the robots: the button that asks for it, and what to do next. */
  renderDrive() {
    const operator = this.hud.operator || {};
    const mine = this.driveLocal;
    const taken = operator.driver != null && !mine;               // somebody else has the robots
    const waiting = Boolean(this.driving?.waiting) && !taken;
    const turn = Number.isFinite(operator.turn_left_s) ? clock(operator.turn_left_s) : '';
    const busy = Boolean(this.hud.copy?.busy);                    // a comparison runs: the runner holds the turn
    // a phone's stage has room for a few words beside the camera buttons
    const ask = taken ? (this.phone.matches ? `In use${turn ? ` · ${turn}` : ''}` : `Somebody is driving${turn ? ` · free in ${turn}` : ''}`)
      : waiting ? 'Starting' : 'Drive it yourself';
    for (const ref of ['drive-stage', 'drive-panel']) {
      this.text(ref, ref === 'drive-panel' && taken ? 'Somebody is driving' : ask);
      this.attr(ref, 'disabled', taken || waiting || busy);
      this.attr(ref, 'title', busy ? 'Held until the comparison ends' : taken ? 'Somebody else has the robots for now' : null);
      this.show(ref, !mine);
    }
    this.show('drive-tools', this.hud.mode !== 'policy' && !('opening' in this.app.dataset));
    // the comparison: one scene with Kaiku off, for whoever wants to see what it is for
    const copy = this.hud.copy || {};
    this.show('copy-stage', Boolean(copy.free) && !mine && !taken && !copy.on && !copy.again);   // the banner says the rest
    this.text('copy-stage', copy.next ? 'Starting' : 'See it without Kaiku');
    this.attr('copy-stage', 'disabled', Boolean(copy.next));
    this.show('drive-back', mine);
    this.show('drive-panel-back', mine);
    this.show('drive-card', mine);
    if (!mine) return;

    const camera = Boolean(this.driving?.camera);
    this.text('drive-camera', camera ? 'Use the pointer' : 'Use my hand');
    // Their own grasp needs their fingers: a camera that sees the hand, or a glove on it.
    // A slow device's hand tracking skips a beat now and then: the words wait two seconds before they change.
    if (operator.fingers) this.fingersSeen = { value: operator.fingers, closure: operator.closure, at: performance.now() };
    const kept = this.fingersSeen && performance.now() - this.fingersSeen.at < 2000 ? this.fingersSeen : null;
    const fingers = operator.fingers || (camera && kept ? kept.value : null);
    const closure = operator.closure || (fingers && kept ? kept.closure : null);
    const own = Boolean(operator.own);
    this.show('drive-own', Boolean(fingers) || camera || Boolean(this.driving?.own));
    this.text('drive-own', this.driving?.own ? 'Help me grasp' : 'Grasp it myself');
    this.attr('drive-own', 'aria-pressed', this.driving?.own ? 'true' : 'false');
    this.text('drive-left', turn ? `${turn} left` : '');
    const robots = this.hud.echo?.robots ?? 0;
    const failed = this.hud.echo?.failed ?? 0;
    const table = operator.table;
    const held = table?.parts.find((part) => part.held);
    const phase = this.hud.mode === 'visitor' ? operator.phase : 'wait';
    const settling = this.hud.status === 'settling';
    let say;
    const whose = fingers === 'glove' ? 'The glove' : 'Your fingers';
    const coil = operator.input === 'coil' || (fingers === 'glove' && Boolean(operator.glove?.coil));
    // The robot's two fingers, as the person's own index and middle finger close them right now.
    const now = closure ? `: index ${Math.round(closure.index * 10) * 10}% closed, middle ${Math.round(closure.middle * 10) * 10}%` : '';
    let more = own ? `${fingers === 'glove' ? 'The glove does' : 'Your fingers do'} the grasp${now}. The robot only turns its wrist${coil ? '' : ' and lifts'}.`
      : fingers ? `${whose} move${fingers === 'glove' ? 's' : ''} the robot's${now ? '' : ' fingers'}${now}. The grasp is assisted.`
        : `${robots} robots redo every move on their own tables.`;
    if (phase === 'wait' || !table) {
      say = 'Your turn starts with the next scene';
    } else if (coil && !operator.placed) {
      say = 'Hold your gloved hand flat and still';
      more = 'Where you hold it is the middle of the table.';
    } else if (own && !settling) {
      const near = coil ? true : this.driving?.nearest?.();
      const press = coil ? 'Lower your hand onto a part' : camera ? 'Over it: push your hand towards the camera' : 'Over it: hold the button down';
      say = held ? (operator.over && operator.over === held.bin ? 'Open your hand' : `Take it to the ${held.color} bin`)
        : near ? `${press}, then close your fingers` : 'Move over a part';
    } else if (settling) {
      say = 'Sorted';
      more = `Now the ${robots} others finish theirs. Every one that gets it right is a demonstration.`;
    } else if (phase === 'pick') {
      say = camera || fingers === 'glove' ? 'Keep pinching' : 'Keep holding';
      more = 'The hand goes down and lifts it.';
    } else if (phase === 'carry' && held) {
      const over = operator.over && operator.over === held.bin;
      say = over ? (camera || fingers === 'glove' ? 'Open your hand' : 'Let go') : `Take it to the ${held.color} bin`;
    } else if (phase === 'release') {
      say = 'In it goes';
    } else if (fingers === 'glove') {
      say = coil ? 'Move your hand over a part and pinch' : 'Point at a part and pinch';
    } else {
      say = camera ? 'Move your hand over a part and pinch' : 'Press on a part and hold';
    }
    if (failed > 0 && !settling) more = `${failed} of ${robots} robots could not follow that. The rest keep up with you.`;
    this.text('drive-say', say);
    this.text('drive-more', more);
  }

  onOwn() {
    if (!this.driving) return;
    this.driving.own = !this.driving.own;
    this.driving.at = this.driving.at || { x: 0.28, y: 0 };
    this.driving.dirty = true;
    this.renderDrive();
  }

  setDriveActive(active) {
    this.driveLocal = Boolean(active);
    this.app.toggleAttribute('data-driving', this.driveLocal);
    this.renderDrive();
  }

  renderEpisode() {
    const episode = this.hud.episode || {};
    const parts = episode.parts || 0;
    const sorted = Math.min(episode.sorted || 0, parts);
    const alone = this.hud.mode === 'policy';
    this.text('scene-word', alone ? 'Round' : 'Scene');
    this.text('scene-index', episode.index != null
      ? (Number.isFinite(episode.target) ? `${episode.index} of ${episode.target}` : String(episode.index)) : '');
    // The clock counts simulated seconds. A host that simulates slower than real time says so beside it.
    const slow = speed(this.hud.perf, '');
    this.text('elapsed', `${clock(episode.elapsed_s || 0)}${slow ? ` · ${slow}` : ''}`);
    this.attr('elapsed', 'title', slow ? 'Simulated time. The simulation runs slower than a clock on this machine.' : null);
    this.text('sorted-text', parts ? `${sorted} of ${parts} parts sorted` : 'No parts on the table yet');

    const pips = this.el.pips;
    const dots = alone ? 0 : parts;          // a round is every robot's parts: too many to draw one by one
    while (pips.children.length < dots) pips.append(document.createElement('i'));
    while (pips.children.length > dots) pips.lastChild.remove();
    for (let i = 0; i < dots; i++) pips.children[i].toggleAttribute('data-on', i < sorted);
    this.show('steps', !alone);

    const step = episode.step;
    const at = STEP_ORDER.indexOf(step);
    const key = `${step}`;
    if (this.written.get('steps') !== key) {
      this.written.set('steps', key);
      for (const item of this.el.steps.children) {
        const i = STEP_ORDER.indexOf(item.dataset.step);
        item.toggleAttribute('data-done', at >= 0 && i < at);
        item.toggleAttribute('data-now', i === at);
        if (i === at) item.setAttribute('aria-current', 'step');
        else item.removeAttribute('aria-current');
      }
    }
    this.text('step-now', alone ? 'Every robot is sorting a scene it has never seen' : STEP_TEXT[step] || '');
  }

  renderHistory() {
    // Scenes an operator drove and rounds the policy sorted alone are two different things: the row
    // shows the kind that is running now, and says which.
    const alone = this.hud.mode === 'policy';
    const all = Array.isArray(this.hud.history) ? this.hud.history : [];
    const history = all.filter((entry) => Boolean(entry.policy) === alone).slice(-40);
    const last = history[history.length - 1];
    const key = `${alone}:${history.length}:${last?.episode}:${last?.ok}`;
    if (key === this.historyKey) return;
    this.historyKey = key;

    this.text('history-title', alone ? 'Last rounds, nobody driving' : 'Last scenes, with an operator');
    this.text('history-empty', alone ? 'The first round is still running.' : 'The first scene is still running.');
    this.attr('history', 'aria-label', alone
      ? 'Robots that sorted their scene alone in each of the last rounds'
      : 'Echoes that succeeded in each of the last scenes');
    this.show('history-empty', history.length === 0);
    this.show('history', history.length > 0);
    const bars = history.map((entry) => {
      const bar = document.createElement('i');
      const robots = entry.robots || 0;
      const share = robots ? entry.ok / robots : 0;
      bar.style.setProperty('--ok-share', `${Math.round(share * 100)}%`);
      bar.toggleAttribute('data-full', share >= 1);
      bar.toggleAttribute('data-lead-failed', entry.lead_ok === false);
      bar.toggleAttribute('data-policy', alone);
      bar.title = alone
        ? `Round ${entry.episode}, nobody driving: ${entry.ok} of ${robots} robots sorted their scene`
        : `Scene ${entry.episode}: ${entry.ok} of ${robots} echoes sorted it`
          + (entry.lead_ok === false ? ', the operator did not' : '');
      return bar;
    });
    this.el.history.replaceChildren(...bars);
    this.text('history-note', !last ? '' : alone
      ? `Round ${last.episode}: ${last.ok} of ${last.robots} alone`
      : `Scene ${last.episode}: ${last.ok} of ${last.robots} echoes`);
  }

  buildMinimap(layout) {
    const map = this.el.minimap;
    map.replaceChildren();
    this.minimap = new Array(this.nEnvs);
    const button = (env) => {
      const cell = document.createElement('button');
      cell.type = 'button';
      cell.dataset.env = String(env);
      cell.setAttribute('aria-label', env === 0 ? "The operator's robot" : `Robot ${env + 1}`);
      cell.title = env === 0 ? "The operator's robot" : `Robot ${env + 1}`;
      this.minimap[env] = cell;
      return cell;
    };
    // drawn as the stage is seen from the front: the last rank on top, the operator's robot below
    let env = 1 + layout.ranks.reduce((a, b) => a + b, 0);
    for (let r = layout.ranks.length - 1; r >= 0; r--) {
      const row = document.createElement('div');
      row.className = 'minimap-rank';
      env -= layout.ranks[r];
      for (let i = 0; i < layout.ranks[r]; i++) row.append(button(env + i));
      map.append(row);
    }
    const lead = document.createElement('div');
    lead.className = 'minimap-rank minimap-lead';
    lead.append(button(0));
    map.append(lead);
    map.addEventListener('click', (event) => {
      const cell = event.target.closest('[data-env]');
      if (cell) this.onFocus(Number(cell.dataset.env) === this.focus ? -1 : Number(cell.dataset.env));
    });
  }

  renderFleet() {
    const { echo = {} } = this.hud;
    if (this.states) {
      for (let e = 0; e < this.minimap.length; e++) {
        const name = STATE_NAMES[this.states[e]] || 'idle';
        const cell = this.minimap[e];
        if (cell.dataset.state !== name) cell.dataset.state = name;
      }
    }
    const robots = echo.robots ?? 0;
    this.text('fleet-note', !robots ? '' : echo.done > 0 ? `${echo.ok ?? 0} of ${robots} sorted` : `${robots} working`);
    this.show('lag-pair', this.hud.mode !== 'policy');
    this.text('lag-mean', seconds(echo.lag_s));
    this.text('lag-max', seconds(echo.lag_max_s));
  }

  renderDataset() {
    const dataset = this.hud.dataset || {};
    const recording = Boolean(dataset.recording);
    this.attr('record', 'aria-checked', String(recording));
    this.attr('record', 'disabled', Boolean(this.hud.shared));       // one session for everybody: it stays on
    this.text('record-note', recording
      ? 'Every robot that sorts its whole scene with no unsafe command is saved as one demonstration.'
      : 'Recording is off. Robots still sort, and nothing is written to disk.');
    this.text('ds-episodes', count(dataset.episodes || 0));
    this.text('ds-frames', count(dataset.frames || 0));
    const minutes = (dataset.frames || 0) / 30 / 60;
    this.text('ds-time', minutes >= 90 ? `${(minutes / 60).toFixed(1)} h` : `${Math.round(minutes)} min`);
    this.show('ds-get', Boolean(dataset.download));
    this.show('ds-where', !this.hud.shared);                          // a path on the server says nothing to a visitor
    this.text('ds-path', dataset.path || 'Not recording to disk');
    this.show('ds-copy', Boolean(dataset.path));
  }

  renderPolicy(live) {
    const policy = live.state && live.state !== 'none' ? live : this.flags.sample ? this.samplePolicy : live;
    const curve = Array.isArray(policy?.curve) ? policy.curve : [];
    const state = policy?.state || 'none';
    const study = policy?.study;
    const key = `${state}:${curve.length}:${curve[curve.length - 1]?.kaiku}`;
    if (key !== this.chartKey) {
      this.chartKey = key;
      // the point where the echoes change the most
      let widest = null;
      for (const point of curve) {
        if (!Number.isFinite(point.kaiku) || !Number.isFinite(point.plain)) continue;
        if (!widest || point.kaiku - point.plain > widest.kaiku - widest.plain) widest = point;
      }
      drawChart(this.el.chart, curve, widest?.demos);
      this.show('chart-empty', curve.length === 0);
      // the same finding where a visitor lands, a click away from the chart
      this.show('headline', Boolean(widest));
      if (widest) {
        this.text('headline-top', `The study: trained on ${widest.demos} demonstrations`);
        this.text('headline-main', `${cap(percent(widest.plain))} alone, ${percent(widest.kaiku)} with echoes`);
        drawSpark(this.el['headline-spark'], curve, widest.demos);
      }
      const best = curve[curve.length - 1];
      const at = Number.isFinite(best?.demos) && curve.length > 1 ? ` at ${best.demos}` : '';
      this.text('policy-kaiku', Number.isFinite(best?.kaiku) ? `${percent(best.kaiku)}${at}` : '');
      this.text('policy-plain', Number.isFinite(best?.plain) ? `${percent(best.plain)}${at}` : '');
      this.show('finding', Boolean(widest));
      if (widest) {
        // ... and the fewest demonstrations that, echoed, beat the most the operator gave alone
        const most = curve[curve.length - 1];
        const enough = curve.find((point) => point.kaiku > most.plain && point.demos < most.demos);
        this.text('finding', `From ${widest.demos} demonstrations by the operator: `
          + `${percent(widest.plain)} alone${spread(widest.plain_runs)}, `
          + `${percent(widest.kaiku)} with echoes${spread(widest.kaiku_runs)}.`
          + (enough ? ` ${enough.demos} with echoes beat ${most.demos} without.` : ''));
      }
      this.show('study-note', Boolean(study?.runs && study?.scenes));
      if (study?.runs && study?.scenes) {
        const ranged = curve.some((point) => Array.isArray(point.plain_runs));
        this.text('study-note', `Each point is the mean of ${study.runs} training runs, tested on ${study.scenes} scenes neither policy had seen.`
          + (ranged ? ' The pale bars run from the lowest run to the highest.' : ''));
      }
    }
    this.show('policy-badge', Boolean(policy?.sample));
    this.renderBaselines(policy?.baselines);

    const success = Number.isFinite(policy?.success) ? policy.success : null;
    const driving = this.hud.mode === 'policy';
    const has = success != null && state !== 'none';
    const held = this.heldBy();
    this.show('policy-now', has);
    if (has) {
      this.text('drive', driving ? 'Take back control' : 'Let it drive');
      this.attr('drive', 'disabled', Boolean(held));
      this.text('drive-note', driving
        ? 'It is driving every robot now. Nobody is wearing gloves.'
        : 'Hand every robot to the policy. Each one gets a scene it has never seen.');
      const echoes = Math.max(0, (policy.demos || 0) - (policy.humans || 0));
      // Which policy this is, so its numbers are not taken for the study's.
      this.text('policy-origin', policy.origin === 'session' ? 'Trained in this session'
        : policy.origin === 'shipped' ? 'Shipped with Kaiku' : policy.origin === 'file' ? 'A policy file' : 'This session');
      this.text('policy-data', Number.isFinite(policy.humans)
        ? `${count(policy.humans)} by the operator, ${count(echoes)} echoes` : 'Demonstrations from this session');
      this.text('policy-test', `${percent(success)} of ${policy.tested_on || 0} new scenes`);
      this.text('policy-alone', Number.isFinite(policy.plain) ? percent(policy.plain) : 'Not trained');
      const tally = policy.alone || {};
      this.text('policy-here', tally.scenes ? `${count(tally.ok)} of ${count(tally.scenes)} scenes sorted` : 'Not run yet');
      this.show('policy-which', Boolean(study?.runs && study?.scenes));
      if (study?.runs && study?.scenes) {
        this.text('policy-which', `One policy from one training run, tested once. The study above is ${study.runs} runs `
          + `at every size on ${study.scenes} scenes, so its numbers are not these.`);
      }
    }

    const training = state === 'training';
    const progress = Math.min(1, Math.max(0, policy?.progress || 0));
    this.show('train-meter', training);
    if (training) this.el['train-fill'].style.transform = `scaleX(${progress})`;
    this.attr('train', 'disabled', training || held?.why === 'visitor');
    this.text('train', training ? 'Training' : 'Train both');
    const eta = Number.isFinite(policy?.eta_s) && policy.eta_s > 0 ? ` About ${left(policy.eta_s)} left.` : '';
    const long = takes(policy);
    const trained = policy?.trained;
    this.text('train-note', training
      ? `${STAGE_TEXT[policy.stage] || 'Training'}, ${where(policy)}: ${Math.round(progress * 100)}%.${eta}`
        + (eta || !long ? '' : ` A run takes ${long}.`)
      : policy?.error ? policy.error
        : 'Trains two policies on the last 16 demonstrations of this session, one with their echoes and one without, '
          + `then tests both on 48 scenes neither has seen.${long ? ` Takes ${long}.` : ''}`);
    // What the last run here came to, kept apart from the study: other demonstrations, fewer test scenes, one run.
    this.show('train-last', Boolean(trained) && !training);
    if (trained && !training) {
      const minutes = Number.isFinite(trained.minutes) && trained.minutes >= 1 ? `, in ${Math.round(trained.minutes)} minutes` : '';
      this.text('train-last', `This session's last training: ${trained.humans || 0} demonstrations, `
        + `tested on ${trained.scenes || 0} new scenes, one run${minutes}. `
        + `${cap(percent(trained.kaiku))} with echoes, ${percent(trained.plain)} alone.`
        + (trained.in_use ? '' : ' The policy that was already here sorts more, so it keeps driving.')
        + (study?.runs && study?.scenes ? ` The study is ${study.runs} runs on ${study.scenes} scenes.` : ''));
    }
  }

  /** The measured comparison with replayed demonstrations: three ways to train from the same few. */
  renderBaselines(measured) {
    const arms = measured?.arms;
    const ok = Boolean(arms?.kaiku && arms?.plain && arms?.mimicgen);
    this.show('baselines', ok);
    const key = ok ? JSON.stringify(measured) : '';
    if (!ok || key === this.baselinesKey) return;
    this.baselinesKey = key;
    const { demos, tables, runs, made = {} } = measured;
    this.text('baselines-lede', `A second measurement, on another ${demos} demonstrations. Three ways to train from them, `
      + `each tested on ${tables} tables it had not seen, ${runs} runs each.`);
    const rows = [
      ['kaiku', "With all of Kaiku's echoes"],
      ['plain', `The ${demos} alone`],
      ['mimicgen', 'With MimicGen-style replays'],
    ];
    this.el['baselines-bars'].replaceChildren(...rows.map(([name, label]) => {
      const arm = arms[name];
      const item = document.createElement('li');
      item.className = name === 'kaiku' ? 'bar bar-kaiku' : 'bar';
      const text = document.createElement('span');
      text.className = 'bar-name';
      text.textContent = label;
      const value = document.createElement('b');
      value.className = 'num';
      value.textContent = `${(arm.yield * 100).toFixed(1)}%`;
      const track = document.createElement('span');
      track.className = 'bar-track';
      const fill = document.createElement('i');
      fill.style.transform = `scaleX(${Math.min(1, Math.max(0, arm.yield))})`;
      track.append(fill);
      const range = Number.isFinite(arm.low) && Number.isFinite(arm.high)
        ? `${Math.round(arm.low * 100)} to ${Math.round(arm.high * 100)}% across the ${runs} runs` : '';
      if (range) {                                            // the lowest run to the highest, drawn on the bar
        const whisker = document.createElement('i');
        whisker.className = 'bar-range';
        whisker.style.left = `${arm.low * 100}%`;
        whisker.style.width = `${Math.max(0, arm.high - arm.low) * 100}%`;
        track.append(whisker);
      }
      item.title = `${arm.clean} of ${arm.scenes} test tables sorted${range ? `, ${range}` : ''}`;
      item.append(text, value, track);
      if (range && arm.high - arm.low >= 0.1) {               // a wide spread is said, not only drawn
        const wide = document.createElement('span');
        wide.className = 'bar-spread';
        wide.textContent = cap(range);
        item.append(wide);
      }
      return item;
    }));
    const replay = made.mimicgen;
    const echo = made.kaiku;
    this.text('baselines-note', "The replay is our own MimicGen-style re-implementation, not MimicGen's code."
      + (replay && echo ? ` On the same ${count(replay.attempts)} tables it made ${count(replay.clean)} clean demonstrations, `
        + `Kaiku ${count(echo.clean)}.` : ''));
  }

  renderControls() {
    const paused = this.hud.status === 'paused';
    this.text('pause-text', paused ? 'Resume' : 'Pause');
    this.attr('pause-icon', 'href', paused ? '#i-play' : '#i-pause');
    this.attr('pause', 'aria-pressed', String(paused));
    // What the runner would refuse right now looks held, and one line says why and for how long.
    const held = this.heldBy();
    this.attr('pause', 'disabled', Boolean(held) && !paused);     // resuming is never held
    this.attr('next', 'disabled', Boolean(held));
    for (const button of this.el.parts.children) if (button.disabled !== Boolean(held)) button.disabled = Boolean(held);
    this.show('held', Boolean(held));
    if (held) this.text('held', held.text);
  }

  // One line at the top of the stage for whatever most needs saying, or nothing.
  renderBanner() {
    const { hud } = this;
    const operator = hud.operator;
    let banner = null;
    if (this.connection === 'reconnecting') {
      banner = { level: 'warn', text: 'Connection lost. Reconnecting', action: 'Retry now', run: () => this.onRetry() };
    } else if (this.connection === 'connecting') {
      banner = { level: 'warn', text: 'Connecting to the runner' };
    } else if (this.stalled) {
      banner = { level: 'warn', text: 'The runner has stopped sending. Waiting for it' };
    } else if (hud.status === 'paused') {
      banner = { level: 'info', text: 'Paused', action: 'Resume', run: () => this.togglePause() };
    } else if (hud.copy?.on && hud.echo?.robots) {
      banner = {
        level: 'warn',
        text: hud.copy.over
          ? `Without Kaiku, ${hud.echo.ok ?? 0} of ${hud.echo.robots} robots sorted their table`
          : `Kaiku is off. All ${hud.echo.robots} robots get the operator's joint commands`,
      };
    } else if (hud.copy?.again && hud.copy.last && hud.echo?.robots) {
      const { copy: without, robots } = hud.copy.last;
      banner = {
        level: 'ok',
        text: hud.copy.over
          ? `With Kaiku, ${hud.echo.ok ?? 0} of ${hud.echo.robots} robots sorted their table. Without: ${without}`
          : `The same tables, Kaiku on. Without it: ${without} of ${robots}`,
      };
    } else if (hud.policy?.state === 'training') {
      // Training takes minutes and its panel is two clicks away: say it where everybody looks.
      const { progress = 0, eta_s: eta } = hud.policy;
      const time = Number.isFinite(eta) && eta > 0 ? `, about ${left(eta)} left` : '';
      // Somebody driving has the table to look at: one short line, and nothing to press by mistake.
      banner = this.driveLocal
        ? { level: 'info', text: `Training ${Math.round(progress * 100)}%${time}` }
        : { level: 'info', text: `Training two policies: ${Math.round(progress * 100)}%${time}`,
            action: 'Show', run: () => this.selectTab('policy') };
    } else if (operator && (operator.message || operator.source === 'gloves' && !(operator.left?.connected && operator.right?.connected))) {
      banner = {
        level: 'warn', text: operator.message || 'No gloves connected', action: 'Use autopilot',
        run: () => this.send({ type: 'cmd', name: 'mode', value: 'autopilot' }),
      };
    } else if (hud.episode?.step === 'calibrating') {
      // only somebody wearing gloves has hands to hold still
      banner = { level: 'info', text: this.drive() === 'gloves' ? 'Calibrating. Hold both hands flat and still'
        : 'The emulated gloves are calibrating' };
    } else if (hud.status === 'settling' && hud.echo?.robots && hud.episode?.end !== 'cut' && !this.driveLocal) {
      // While the echoes are still finishing the count grows; "sorted" is said once, when the scene has ended.
      const { ok = 0, robots } = hud.echo;
      const ended = hud.episode?.end === 'sorted' || hud.episode?.end === undefined;
      banner = {
        level: 'ok',
        text: hud.mode === 'policy'
          ? `${ok} of ${robots} robots sorted their scene with nobody driving`
          : ended ? `Scene ${hud.episode?.index ?? ''} sorted by ${ok} of ${robots} echoes`
            : `Scene ${hud.episode?.index ?? ''}: ${ok} of ${robots} echoes have sorted it so far`,
      };
    }

    this.bannerRun = banner?.run || null;
    this.show('banner', Boolean(banner));
    if (!banner) return;
    this.attr('banner', 'data-level', banner.level);
    this.text('banner-text', banner.text);
    this.show('banner-action', Boolean(banner.action));
    if (banner.action) this.text('banner-action', banner.action);
  }

  // ------------------------------------------------------------------ camera and focus

  setShot(shot) {
    for (const button of this.el.shots.children) button.setAttribute('aria-checked', String(button.dataset.shot === shot));
  }

  setFocus(env) {
    if (this.focus >= 0) this.minimap[this.focus]?.removeAttribute('data-focus');
    this.focus = env;
    if (env >= 0) this.minimap[env]?.setAttribute('data-focus', '');
    this.show('focus', env >= 0);
    this.renderFocus();
  }

  renderFocus() {
    const env = this.focus;
    if (env < 0) return;
    const { envs = {}, episode = {} } = this.hud;
    const state = this.states ? this.states[env] : IDLE;
    this.text('focus-state', STATE_TEXT[state] || STATE_TEXT[IDLE]);
    this.attr('focus-dot', 'data-state', STATE_NAMES[state] || 'idle');
    const alone = this.hud.mode === 'policy';
    const parts = envs.parts?.[env] ?? (alone ? 0 : episode.parts || 0);
    this.text('focus-title', env === 0 && !alone ? "The operator's robot" : `Robot ${env + 1}`);
    this.text('focus-sorted', parts ? `${Math.min(envs.sorted?.[env] ?? 0, parts)} of ${parts}` : 'None yet');
    this.text('focus-lag', alone ? 'No operator' : env === 0 ? 'Leads' : seconds(envs.lag_s?.[env]));
    this.text('focus-clipped', count(envs.clipped?.[env] || 0));
  }

  // ------------------------------------------------------------------ every frame

  /**
   * @param {number} now  performance.now()
   * @param {{x: number, y: number, visible: boolean}} tag  where the operator's caption goes, in stage pixels
   */
  frame(now, dt, tag) {
    const counter = this.demos;
    if (counter.shown !== counter.goal) {
      counter.shown += (counter.goal - counter.shown) * (1 - Math.exp(-dt * 5));
      if (Math.abs(counter.goal - counter.shown) < 0.5) counter.shown = counter.goal;
    }
    this.text('demos', count(Math.round(counter.shown)));
    this.attr('demos', 'data-tick', now < counter.tickUntil);
    this.attr('gain', 'data-show', now < counter.gainUntil);

    const shown = this.tag;
    if (tag.visible !== shown.visible) {
      shown.visible = tag.visible;
      this.el.tag.hidden = !tag.visible;
    }
    if (tag.visible && (Math.abs(tag.x - shown.x) > 0.25 || Math.abs(tag.y - shown.y) > 0.25)) {
      shown.x = tag.x;
      shown.y = tag.y;
      this.el.tag.style.transform = `translate(${tag.x.toFixed(1)}px, ${tag.y.toFixed(1)}px) translate(-50%, -100%)`;
    }
  }

  // ------------------------------------------------------------------ loading and messages

  /** Loading screen: what is being fetched and how far along (0..1). */
  loading(step, fraction) {
    this.attr('veil', 'data-error', false);
    this.text('veil-step', step);
    this.text('veil-percent', `${Math.round(Math.min(1, Math.max(0, fraction)) * 100)}%`);
    this.attr('veil', 'data-known', true);                           // from here on the bar shows how far it is
    this.el['veil-fill'].style.transform = `scaleX(${Math.min(1, Math.max(0.04, fraction))})`;
    this.attr('veil-bar', 'aria-valuenow', String(Math.round(fraction * 100)));
  }

  loadingFailed(step) {
    this.attr('veil', 'data-error', true);
    this.text('veil-step', step);
    this.text('veil-percent', '');
    this.el['veil-retry'].hidden = false;
  }

  loaded() {
    this.attr('veil', 'data-done', true);
  }

  /** The opening's three lines, for whoever is driving right now. */
  openingWords(robots) {
    if (this.hud.mode === 'policy') {
      return ['Nobody is driving this robot.', `${robots} robots. Nobody driving.`, 'Every scene is new to them.',
        'They learned it from the echoes.'];
    }
    // Said from what is driving right now, never from what could be: no gloves unless gloves are the source.
    const name = this.drive();
    const drive = DRIVES[name];
    const first = !drive ? 'One robot is being driven.'
      : name === 'stand-in' ? 'A scripted hand in emulated gloves drives this robot.'
        : name === 'gloves' ? 'Rokoko gloves drive this robot.'
          : `${drive.name} ${drive.verb || 'drives'} this robot.`;
    const one = name === 'gloves' ? 'One pair of gloves.'
      : name === 'pointer' ? 'One pointer.'
        : name === 'camera' ? 'One hand.'
          : name?.startsWith('glove-') ? 'One glove.'
            : name === 'autopilot' ? 'One script.' : 'One operator.';
    return [first, `${one} ${robots} robots.`, 'Every table is different.', `Do it once. Robots learn from ${robots}.`];
  }

  toast(text, level = 'info') {
    if (!text) return;
    const list = this.el.toasts;
    // the same message again only restarts its clock
    const same = [...list.children].find((item) => item.textContent === text && !item.dataset.leaving);
    if (same) {
      clearTimeout(same.timer);
      same.timer = setTimeout(() => this.dismiss(same), TOAST_MS);
      return;
    }
    const item = document.createElement('div');
    item.className = 'toast';
    item.dataset.level = level;
    item.textContent = text;
    list.append(item);
    while (list.children.length > MAX_TOASTS) list.firstChild.remove();
    item.timer = setTimeout(() => this.dismiss(item), TOAST_MS);
  }

  dismiss(item) {
    item.dataset.leaving = '';
    setTimeout(() => item.remove(), 260);
  }
}

function count(n) {
  return Math.round(n).toLocaleString('en-US');
}

/** A sentence that starts with a number's words: "under 1% alone" is "Under 1% alone" at its head. */
function cap(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** The spread of a result across its training runs, said only when it is wide: ", 40 to 71% across 3 runs". */
function spread(runs) {
  if (!Array.isArray(runs) || runs.length < 2) return '';
  const low = Math.min(...runs);
  const high = Math.max(...runs);
  if (!(high - low >= 0.1)) return '';
  return ` (${Math.round(low * 100)} to ${Math.round(high * 100)}% across ${runs.length} runs)`;
}

/** How fast the simulation runs against a clock, when it is not real time: "0.8× real time". */
function speed(perf, lead) {
  const rate = perf?.realtime;
  if (!Number.isFinite(rate) || rate <= 0 || rate >= 0.9) return '';
  return `${lead ? `${lead} at ` : ''}${rate.toFixed(1)}× real time`;
}

function trim(n) {
  return n >= 100 ? String(Math.round(n)) : n.toFixed(1).replace(/\.0$/, '');
}

function seconds(value) {
  return `${(Number.isFinite(value) ? value : 0).toFixed(1)} s`;
}

function clock(s) {
  const whole = Math.max(0, Math.floor(s));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

function sentence(word) {
  return word.charAt(0).toUpperCase() + word.slice(1).replace(/[_-]/g, ' ');
}
