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
export function previewHud(hud, name) {
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
    case 'training':
      // a run a third of the way through, with the time left as the runner would say it
      d.policy = {
        ...d.policy, state: 'training', stage: 'training', progress: 0.36, loss: 0.0631, last: 16,
        about_min: d.policy?.about_min ?? 25, eta_s: Math.round((d.policy?.about_min ?? 25) * 60 * 0.64),
      };
      break;
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
      d.operator = { source: 'visitor', drive: 'pointer', input: 'pointer', fingers: null, own: false, driver: -1, turn_left_s: 205, hands: null };
      break;
    case 'comparing':
      d.copy = { ...d.copy, on: true, busy: true, free: false };
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
      // the Train button where it trains on rented GPUs
      d.policy = { ...d.policy, rented: true, about_min: 4 };
      break;
    case 'partial':
      // a runner that only knows the basics
      return { mode: d.mode, status: d.status };
    default:
      break;
  }
  return d;
}
