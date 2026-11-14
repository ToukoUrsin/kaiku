// The policy chart: sorting success against the number of operator demonstrations, one line with
// Kaiku's echoes and one without. Drawn as plain SVG into a 320 x 196 viewBox.

const NS = 'http://www.w3.org/2000/svg';

/** A share as a person reads it. A small rate that is not nothing is never rounded to 0%, nor a near miss to 100%. */
export function percent(share) {
  const value = share * 100;
  if (value > 0 && value < 1) return 'under 1%';
  if (value < 100 && value > 99.5) return 'over 99%';
  return `${Math.round(value)}%`;
}

/** The lowest and highest of a point's training runs, or null when it has one run or none. */
function range(point, series) {
  const runs = point[`${series}_runs`];
  if (!Array.isArray(runs) || runs.length < 2) return null;
  return { low: Math.min(...runs), high: Math.max(...runs), runs: runs.length };
}
const LEFT = 36, RIGHT = 306, TOP = 10, BOTTOM = 168;

/**
 * @param {SVGSVGElement} svg
 * @param {{demos: number, plain: number, kaiku: number, plain_runs?: number[], kaiku_runs?: number[]}[]} curve
 *   success as 0..1, and each training run's where they are known; empty draws the bare axes
 * @param {number} [mark]  demonstrations to mark: the gap between the two lines there is labelled
 */
export function drawChart(svg, curve, mark) {
  svg.replaceChildren();
  const points = curve.filter((p) => Number.isFinite(p?.demos)).sort((a, b) => a.demos - b.demos);

  for (const share of [0, 0.5, 1]) {
    const y = yOf(share);
    svg.append(node('line', { class: share === 0 ? 'axis' : 'grid', x1: LEFT, x2: RIGHT, y1: y, y2: y }));
    svg.append(node('text', { x: LEFT - 8, y: y + 3.5, 'text-anchor': 'end' }, `${share * 100}%`));
  }
  svg.append(node('text', { x: (LEFT + RIGHT) / 2, y: 194, 'text-anchor': 'middle' }, 'Operator demonstrations'));
  if (points.length === 0) return;

  // demonstrations double from point to point, so space them evenly rather than by value
  const xOf = (i) => (points.length === 1 ? (LEFT + RIGHT) / 2 : LEFT + 10 + (i * (RIGHT - LEFT - 20)) / (points.length - 1));
  points.forEach((p, i) => {
    svg.append(node('text', { x: xOf(i), y: BOTTOM + 14, 'text-anchor': 'middle' }, String(p.demos)));
  });

  // the gap that matters, marked where it is widest
  const at = points.findIndex((p) => p.demos === mark);
  const marked = points[at];
  if (marked && Number.isFinite(marked.kaiku) && Number.isFinite(marked.plain)) {
    const x = xOf(at);
    svg.append(node('line', { class: 'gap', x1: x, x2: x, y1: yOf(marked.plain), y2: yOf(marked.kaiku) }));
    svg.append(node('text', { class: 'gap-kaiku', x: x - 8, y: yOf(marked.kaiku) - 6, 'text-anchor': 'end' }, percent(marked.kaiku)));
    svg.append(node('text', { class: 'gap-plain', x: x - 8, y: yOf(marked.plain) - 6, 'text-anchor': 'end' }, percent(marked.plain)));
  }

  for (const series of ['plain', 'kaiku']) {
    const path = points
      .map((p, i) => (Number.isFinite(p[series]) ? `${i === 0 ? 'M' : 'L'}${xOf(i).toFixed(1)} ${yOf(p[series]).toFixed(1)}` : ''))
      .join(' ')
      .replace(/^\s*L/, 'M');
    if (path.trim()) svg.append(node('path', { class: `series series-${series}`, d: path }));
    points.forEach((p, i) => {
      if (!Number.isFinite(p[series])) return;
      // the runs behind the mean: a thin bar from the lowest to the highest, where they differ enough to see
      const runs = range(p, series);
      if (runs && runs.high - runs.low >= 0.03) {
        svg.append(node('line', { class: `runs runs-${series}`, x1: xOf(i), x2: xOf(i), y1: yOf(runs.low), y2: yOf(runs.high) }));
      }
      const dot = node('circle', { class: `dot-${series}`, cx: xOf(i), cy: yOf(p[series]), r: i === at ? 3.6 : 2.6 });
      dot.append(node('title', {}, `${p.demos} demonstrations: ${percent(p[series])}`
        + (runs ? `, ${Math.round(runs.low * 100)} to ${Math.round(runs.high * 100)}% across ${runs.runs} runs` : '')));
      svg.append(dot);
    });
  }
}

/** The two lines alone, small enough for a button: 72 x 34, the marked point with its dots. */
export function drawSpark(svg, curve, mark) {
  svg.replaceChildren();
  const points = curve.filter((p) => Number.isFinite(p?.demos) && Number.isFinite(p.kaiku) && Number.isFinite(p.plain))
    .sort((a, b) => a.demos - b.demos);
  if (points.length < 2) return;
  const x = (i) => 4 + (i * 64) / (points.length - 1);
  const y = (share) => 30 - Math.min(1, Math.max(0, share)) * 26;
  for (const series of ['plain', 'kaiku']) {
    const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)} ${y(p[series]).toFixed(1)}`).join(' ');
    svg.append(node('path', { class: `series series-${series}`, d }));
  }
  const at = points.findIndex((p) => p.demos === mark);
  if (at < 0) return;
  svg.append(node('line', { class: 'gap', x1: x(at), x2: x(at), y1: y(points[at].plain), y2: y(points[at].kaiku) }));
  for (const series of ['plain', 'kaiku']) {
    svg.append(node('circle', { class: `dot-${series}`, cx: x(at), cy: y(points[at][series]), r: 2.6 }));
  }
}

function yOf(share) {
  return BOTTOM - Math.min(1, Math.max(0, share)) * (BOTTOM - TOP);
}

function node(name, attributes, text) {
  const element = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text != null) element.textContent = text;
  return element;
}
