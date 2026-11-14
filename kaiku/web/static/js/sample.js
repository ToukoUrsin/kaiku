// Stand-in numbers for parts of the page the runner cannot show on demand. Nothing here is used
// unless the address asks for it (?sample=1, ?preview=NAME), and the policy tab says so on screen.

const SAMPLE_CURVE = [
  { demos: 1, plain: 0.03, kaiku: 0.41 },
  { demos: 2, plain: 0.08, kaiku: 0.63 },
  { demos: 5, plain: 0.21, kaiku: 0.82 },
  { demos: 10, plain: 0.38, kaiku: 0.91 },
  { demos: 20, plain: 0.57, kaiku: 0.95 },
];

export function samplePolicy() {
  return { state: 'ready', progress: 1, loss: 0.0142, success: 0.95, curve: SAMPLE_CURVE, sample: true };
}

/**
 * Change a HUD message into the state `?preview=NAME` asks for, so every state of the page can be
 * looked at (and photographed) without unplugging anything.
 */
export function previewHud(hud, name, client = null) {
  const d = structuredClone(hud);
  switch (name) {
    case 'paused':
      d.status = 'paused';
      break;
    case 'nogloves':
      d.mode = 'gloves';
      d.operator = {
        source: 'gloves', stand_in: false, calibrated: false, hands: null,
        left: { connected: false, hz: 0, status: 'searching' },
        right: { connected: false, hz: 0, status: 'searching' },
      };
      break;
    case 'gloves':
      d.mode = 'gloves';
      d.operator = { ...d.operator, source: 'gloves', stand_in: false };
      break;
    case 'autopilot':
      d.mode = 'autopilot';
      d.operator = { ...d.operator, source: 'autopilot', stand_in: false };
      break;
    case 'first':
      d.history = [];
      d.totals = { episodes: 0, human_demos: 0, robot_demos: 0, amplification: 0, yield: 0, demos_per_min: 0, clipped_steps: 0, minutes: 0.2 };
      d.dataset = { episodes: 0, frames: 0, path: '', recording: false };
      break;
    case 'training': {
      // a run a third of the way through. The time it takes is the runner's own estimate, or none:
      // a number typed in here could disagree with what the runner says elsewhere on the page.
      const about = d.policy?.about_min;
      d.policy = {
        ...d.policy, state: 'training', stage: 'training', progress: 0.36, loss: 0.0631, last: 16,
        eta_s: Number.isFinite(about) ? Math.round(about * 60 * 0.64) : null,
      };
      break;
    }
    case 'trained':
      // a Train run that ended ten minutes ago
      d.policy = {
        ...d.policy,
        trained: { kaiku: 1, plain: 0.9792, humans: 16, demos: 763, scenes: 48, in_use: true, minutes: 24.6, at: 1, age_s: 600 },
      };
      break;
    case 'visitor':
      // somebody else has the robots, and drives with a pointer
      d.mode = 'visitor';
      d.operator = { source: 'visitor', drive: 'pointer', input: 'pointer', fingers: null, own: false, driver: -1, turn_left_s: 205, queue: [], hands: null };
      break;
    case 'queued':
      // somebody else has the robots, and this page asked to drive next
      d.mode = 'visitor';
      d.operator = { source: 'visitor', drive: 'camera', input: 'camera', fingers: 'camera', own: false, driver: -1, turn_left_s: 132, queue: [client], hands: null };
      break;
    case 'alone':
      // the policy has just been handed the robots: its first round is still running
      d.mode = 'policy';
      d.policy = { ...d.policy, state: 'running', alone: { rounds: 0, scenes: 0, ok: 0, parts: 0, sorted: 0, clipped: 0 } };
      d.echo = { ...d.echo, robots: (d.echo?.robots ?? 7) + 1 };      // with nobody driving, every robot is the policy's
      break;
    case 'comparing':
      // the half without Kaiku runs now; an older comparison's result is still in hand
      d.copy = { ...d.copy, on: true, busy: true, free: false,
        last: { copy: 5, echo: (d.echo?.robots ?? 47) - 1, robots: d.echo?.robots ?? 47, at: 1, age_s: 400 } };
      break;
    case 'compared': {
      const robots = d.echo?.robots ?? 47;
      d.copy = { ...d.copy, last: { copy: 0, echo: robots - 1, robots, at: 1, age_s: 140 } };
      break;
    }
    case 'slow':
      // the hosted simulation, which does not keep up with the clock
      d.perf = { ...d.perf, realtime: 0.7 };
      break;
    case 'rented':
      // the Train button where it trains on rented GPUs. This machine's minutes are not the rented ones,
      // and the page has no other number to say: it says none.
      d.policy = { ...d.policy, rented: true };
      delete d.policy.about_min;
      break;
    case 'partial':
      // a runner that only knows the basics
      return { mode: d.mode, status: d.status };
    default:
      break;
  }
  return d;
}
