// Everything on the page that is not the 3D stage: totals, panels, controls, banners and toasts.
// The runner's HUD message arrives ten times a second; any part of it may be missing, and the
// page must still make sense. Elements are found by their data-ref in index.html.

import { drawChart } from './chart.js';
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
const SOURCE_TEXT = {
  'stand-in': 'Stand-in operator · emulated Smartgloves II',
  gloves: 'Smartgloves II · live',
  autopilot: 'Autopilot · scripted motion',
};
const TAG_TEXT = { 'stand-in': 'Stand-in operator', gloves: 'Operator', autopilot: 'Autopilot' };
const TOAST_MS = 4200;
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

    this.demos = { shown: 0, goal: 0, known: false, tickUntil: 0, gainUntil: 0 };
    this.tag = { x: -1, y: -1, visible: false };

    this.onShot = () => {};
    this.onFocus = () => {};
    this.onRetry = () => {};
    this.onLayout = () => {};
    this.bannerRun = null;

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
      command('mode', button.dataset.mode);
    });
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

    el['focus-close'].addEventListener('click', () => this.onFocus(-1));
    el['focus-prev'].addEventListener('click', () => this.onFocus(this.focus - 1));
    el['focus-next'].addEventListener('click', () => this.onFocus(this.focus + 1));

    el.tabs.addEventListener('click', (event) => {
      const tab = event.target.closest('[data-tab]');
      if (tab) this.selectTab(tab.dataset.tab);
    });
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
  }

  renderTopbar() {
    const { mode, perf } = this.hud;
    for (const button of this.el.mode.children) {
      const checked = button.dataset.mode === mode;
      if (checked) delete button.dataset.pending;
      if (button.getAttribute('aria-checked') !== String(checked)) button.setAttribute('aria-checked', String(checked));
    }
    this.text('perf', Number.isFinite(perf?.realtime) ? `Simulating at ${perf.realtime.toFixed(1)}× real time` : '');
  }

  renderStory() {
    const { totals = {}, echo = {}, dataset = {}, operator = {} } = this.hud;
    const robots = echo.robots ?? Math.max(0, (this.nEnvs || 1) - 1);
    this.text('robots', String(robots));

    const demos = (totals.human_demos || 0) + (totals.robot_demos || 0);
    const counter = this.demos;
    if (!counter.known) {
      counter.shown = demos;
      counter.known = this.hud.totals != null;
    } else if (demos > counter.goal) {
      const now = performance.now();
      counter.tickUntil = now + 900;
      counter.gainUntil = now + 2600;
      this.text('gain', `+${demos - counter.goal}`);
    }
    counter.goal = demos;
    this.text('demos-label', dataset.recording ? 'demonstrations saved' : 'demonstrations');

    const amplification = Number.isFinite(totals.amplification) ? `${trim(totals.amplification)}×` : '0×';
    const yieldText = Number.isFinite(totals.yield) ? `${Math.round(totals.yield * 100)}%` : '0%';
    const rate = Number.isFinite(totals.demos_per_min) ? String(Math.round(totals.demos_per_min)) : '0';
    const clipped = count(totals.clipped_steps || 0);
    this.text('amplification', amplification);
    this.text('amplification-2', amplification);
    this.text('yield', yieldText);
    this.text('yield-2', yieldText);
    this.text('rate', rate);
    this.text('rate-2', rate);
    this.text('clipped', clipped);
    this.text('clipped-2', clipped);
    this.attr('clipped', 'data-alert', totals.clipped_steps > 0);
    this.attr('clipped-2', 'data-alert', totals.clipped_steps > 0);

    const who = operator.source === 'autopilot'
      ? 'Autopilot drives one robot.'
      : operator.source === 'gloves' ? 'One operator in Smartgloves II.' : 'Stand-in operator in emulated Smartgloves II.';
    this.text('caption', `${who} ${robots} robots repeat every motion, each in its own scene.`);
  }

  renderOperator() {
    const operator = this.hud.operator || {};
    const source = operator.source;
    const linked = Boolean(operator.left?.connected && operator.right?.connected);
    const sourceText = source === 'gloves' && !linked ? 'Smartgloves II · not connected' : SOURCE_TEXT[source];
    this.text('source-text', sourceText || 'Waiting for the operator');
    this.attr('source', 'data-live', source === 'gloves' && linked);
    this.attr('source', 'data-standin', source === 'stand-in' || source === 'autopilot');
    this.text('tag-text', TAG_TEXT[source] || 'Operator');

    for (const side of ['left', 'right']) {
      const glove = operator[side] || {};
      const status = glove.connected ? sentence(glove.status || 'connected') : 'No signal';
      this.text(`${side}-status`, status);
      this.text(`${side}-hz`, glove.connected && Number.isFinite(glove.hz) ? `${Math.round(glove.hz)} Hz` : '');
      this.attr(`${side}-dot`, 'data-state', !glove.connected ? null : glove.status === 'tracking' ? 'tracking' : 'weak');
    }
    this.text('calibrated', operator.calibrated ? 'Calibrated' : 'Not calibrated');
    const calibrating = this.hud.episode?.step === 'calibrating';
    this.text('calibrate', calibrating ? 'Calibrating' : operator.calibrated ? 'Recalibrate' : 'Calibrate');
    this.attr('calibrate', 'disabled', calibrating || source === 'autopilot');
  }

  renderEpisode() {
    const episode = this.hud.episode || {};
    const parts = episode.parts || 0;
    const sorted = Math.min(episode.sorted || 0, parts);
    this.text('scene-index', episode.index != null ? String(episode.index) : '');
    this.text('elapsed', clock(episode.elapsed_s || 0));
    this.text('sorted-text', parts ? `${sorted} of ${parts} parts sorted` : 'No parts on the table yet');

    const pips = this.el.pips;
    while (pips.children.length < parts) pips.append(document.createElement('i'));
    while (pips.children.length > parts) pips.lastChild.remove();
    for (let i = 0; i < parts; i++) pips.children[i].toggleAttribute('data-on', i < sorted);

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
    this.text('step-now', STEP_TEXT[step] || '');
  }

  renderHistory() {
    const history = Array.isArray(this.hud.history) ? this.hud.history.slice(-40) : [];
    const last = history[history.length - 1];
    const key = `${history.length}:${last?.episode}:${last?.ok}`;
    if (key === this.historyKey) return;
    this.historyKey = key;

    this.show('history-empty', history.length === 0);
    this.show('history', history.length > 0);
    const bars = history.map((entry) => {
      const bar = document.createElement('i');
      const robots = entry.robots || 0;
      const share = robots ? entry.ok / robots : 0;
      bar.style.setProperty('--ok-share', `${Math.round(share * 100)}%`);
      bar.toggleAttribute('data-full', share >= 1);
      bar.toggleAttribute('data-lead-failed', entry.lead_ok === false);
      bar.title = `Scene ${entry.episode}: ${entry.ok} of ${robots} echoes sorted it`
        + (entry.lead_ok === false ? ', the operator did not' : '');
      return bar;
    });
    this.el.history.replaceChildren(...bars);
    this.text('history-note', last ? `Scene ${last.episode}: ${last.ok} of ${last.robots} echoes` : '');
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
    this.text('lag-mean', seconds(echo.lag_s));
    this.text('lag-max', seconds(echo.lag_max_s));
  }

  renderDataset() {
    const dataset = this.hud.dataset || {};
    const recording = Boolean(dataset.recording);
    this.attr('record', 'aria-checked', String(recording));
    this.text('record-note', recording
      ? 'Every robot that sorts its whole scene without a clipped command is saved as one demonstration.'
      : 'Recording is off. Robots still sort, and nothing is written to disk.');
    this.text('ds-episodes', count(dataset.episodes || 0));
    this.text('ds-frames', count(dataset.frames || 0));
    const minutes = (dataset.frames || 0) / 30 / 60;
    this.text('ds-time', minutes >= 90 ? `${(minutes / 60).toFixed(1)} h` : `${Math.round(minutes)} min`);
    this.text('ds-path', dataset.path || 'Not recording to disk');
    this.show('ds-copy', Boolean(dataset.path));
  }

  renderPolicy(live) {
    const policy = live.state && live.state !== 'none' ? live : this.flags.sample ? this.samplePolicy : live;
    const curve = Array.isArray(policy?.curve) ? policy.curve : [];
    const state = policy?.state || 'none';
    const key = `${state}:${curve.length}:${curve[curve.length - 1]?.kaiku}`;
    if (key !== this.chartKey) {
      this.chartKey = key;
      drawChart(this.el.chart, curve);
      this.show('chart-empty', curve.length === 0);
      const best = curve[curve.length - 1];
      this.text('policy-kaiku', Number.isFinite(best?.kaiku) ? `${Math.round(best.kaiku * 100)}%` : '');
      this.text('policy-plain', Number.isFinite(best?.plain) ? `${Math.round(best.plain * 100)}%` : '');
    }
    this.show('policy-badge', Boolean(policy?.sample));

    const training = state === 'training';
    const progress = Math.min(1, Math.max(0, policy?.progress || 0));
    this.show('train-meter', training);
    if (training) this.el['train-fill'].style.transform = `scaleX(${progress})`;
    this.attr('train', 'disabled', training);
    this.text('train', training ? 'Training' : state === 'none' ? 'Train both' : 'Train again');
    const loss = Number.isFinite(policy?.loss) ? `, loss ${policy.loss.toFixed(4)}` : '';
    const success = Number.isFinite(policy?.success) ? ` It sorts ${Math.round(policy.success * 100)}% of new scenes.` : '';
    this.text('train-note', {
      none: 'Nothing has been trained in this session. Training uses the demonstrations saved so far.',
      training: `Training both policies: ${Math.round(progress * 100)}%${loss}.`,
      ready: `Both policies are trained.${success}`,
      running: `The policy trained with echoes is driving the robots.${success}`,
    }[state] || '');
  }

  renderControls() {
    const paused = this.hud.status === 'paused';
    this.text('pause-text', paused ? 'Resume' : 'Pause');
    this.attr('pause-icon', 'href', paused ? '#i-play' : '#i-pause');
    this.attr('pause', 'aria-pressed', String(paused));
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
    } else if (operator && operator.source === 'gloves' && !(operator.left?.connected && operator.right?.connected)) {
      banner = {
        level: 'warn', text: 'No gloves connected', action: 'Use autopilot',
        run: () => this.send({ type: 'cmd', name: 'mode', value: 'autopilot' }),
      };
    } else if (hud.episode?.step === 'calibrating') {
      banner = { level: 'info', text: 'Calibrating. Hold both hands flat and still' };
    } else if (hud.status === 'settling' && hud.echo?.robots) {
      banner = { level: 'ok', text: `Scene ${hud.episode?.index ?? ''} sorted by ${hud.echo.ok ?? 0} of ${hud.echo.robots} echoes` };
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
    this.text('focus-title', env === 0 ? "The operator's robot" : `Robot ${env + 1}`);
    this.text('focus-state', STATE_TEXT[state] || STATE_TEXT[IDLE]);
    this.attr('focus-dot', 'data-state', STATE_NAMES[state] || 'idle');
    const parts = episode.parts || 0;
    this.text('focus-sorted', parts ? `${Math.min(envs.sorted?.[env] ?? 0, parts)} of ${parts}` : 'None yet');
    this.text('focus-lag', env === 0 ? 'Leads' : seconds(envs.lag_s?.[env]));
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
    this.el['veil-fill'].style.transform = `scaleX(${Math.min(1, Math.max(0.04, fraction))})`;
    this.attr('veil-bar', 'aria-valuenow', String(Math.round(fraction * 100)));
  }

  loadingFailed(step) {
    this.attr('veil', 'data-error', true);
    this.text('veil-step', step);
  }

  loaded() {
    this.attr('veil', 'data-done', true);
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
