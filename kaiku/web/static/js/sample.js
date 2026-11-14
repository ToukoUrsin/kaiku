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
      d.policy = { state: 'training', progress: 0.42, loss: 0.0631, success: null, curve: SAMPLE_CURVE.slice(0, 2), sample: true };
      break;
    case 'partial':
      // a runner that only knows the basics
      return { mode: d.mode, status: d.status };
    default:
      break;
  }
  return d;
}
