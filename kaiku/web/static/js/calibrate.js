// The short, hands-free calibration card. It only appears for a live, uncalibrated glove pair.
export class CalibrationCard {
  constructor({ app, send }) {
    this.app = app;
    this.send = send;
    this.el = {};
    for (const node of app.querySelectorAll('[data-ref]')) this.el[node.dataset.ref] = node;
  }

  update(hud) {
    const op = hud?.operator || {};
    const connected = Boolean(op.left?.connected || op.right?.connected);
    const live = op.source === 'gloves' && connected && (!op.calibrated || !(op.left?.closed && op.right?.closed));
    this.el['calibrate-card'].hidden = !live;
    if (!live) return;
    const left = op.left || {}, right = op.right || {};
    for (const [side, hand] of [['left', left], ['right', right]]) {
      const node = this.el[`calibrate-${side}`];
      const ok = hand.seen && hand.position && hand.flat;
      node.dataset.state = ok ? 'ok' : 'bad';
      node.lastChild.textContent = !hand.seen ? `${side[0].toUpperCase() + side.slice(1)} hand · no signal`
        : !hand.position ? `${side[0].toUpperCase() + side.slice(1)} hand · no position`
          : !hand.flat ? `${side[0].toUpperCase() + side.slice(1)} hand · not flat`
            : `${side[0].toUpperCase() + side.slice(1)} hand`;
    }
    const held = Math.min(1, Math.max(0, Math.min(left.still_s || 0, right.still_s || 0) / 1.5));
    this.el['calibrate-ring'].style.setProperty('--hold', String(Math.round(held * 100)));
    const closed = Boolean(left.closed && right.closed);
    if (op.calibrated) {
      this.el['calibrate-title'].textContent = closed ? 'Calibrated. The robot follows your hands.' : 'Close each hand once';
      this.el['calibrate-note'].textContent = closed ? '' : 'Close thumb to fingers, one hand at a time.';
    } else {
      this.el['calibrate-title'].textContent = held >= 1 ? 'Release, then close each hand once' : 'Hold both hands flat over the Coil Pro, palms down';
      this.el['calibrate-note'].textContent = held >= 1 ? 'Calibration captured.' : 'Keep still for a moment.';
    }
  }
}
