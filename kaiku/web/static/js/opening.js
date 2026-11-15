// The first seconds of a visit: close on the operator's hands, back to every robot, and two lines
// that say what this is. Any click, key or scroll ends it and gives the page to the visitor.

const KEY = 'kaiku-opened';

/** Whether this visit gets the opening: once per tab, not for deep links, never when motion is unwelcome. */
export function wantsOpening(flags) {
  if (flags.intro === 'on') return true;
  if (flags.intro === 'off' || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return false;
  try {
    if (sessionStorage.getItem(KEY)) return false;
    sessionStorage.setItem(KEY, '1');
  } catch {
    // private windows may refuse storage: play it
  }
  return true;
}

/**
 * @param {object} o
 * @param {HTMLElement} o.app
 * @param {HTMLElement} o.line     where the words go
 * @param {import('./camera.js').CameraRig} o.rig
 * @param {object} o.wide          the shot that shows every robot
 * @param {() => string[]} o.words four lines: on the hands, on every robot, what makes it more than a copy, why it matters
 * @param {() => void} o.rest      puts the camera where the page normally starts
 * @returns {{refresh: () => void}} `refresh` says the line on screen again from `words()`: who drives can
 *   change while a line is up (a visitor's turn starts), and the line must not go on saying who drove before
 */
export function playOpening({ app, line, rig, wide, words, rest }) {
  const timers = [];
  let over = false;
  let showing = -1;                 // which of the lines is up, or -1
  const at = (seconds, run) => timers.push(setTimeout(run, seconds * 1000));
  const say = (index) => {
    line.removeAttribute('data-show');
    showing = -1;
    if (index < 0) return;
    timers.push(setTimeout(() => {
      line.textContent = words()[index];
      line.setAttribute('data-show', '');
      showing = index;
    }, 280));
  };
  const refresh = () => {
    if (over || showing < 0) return;
    const text = words()[showing];
    if (text && line.textContent !== text) line.textContent = text;
  };
  const end = (event) => {
    if (over) return;
    over = true;
    timers.forEach(clearTimeout);
    for (const name of ['pointerdown', 'keydown', 'wheel']) window.removeEventListener(name, end, true);
    line.removeAttribute('data-show');
    const back = () => {
      app.removeAttribute('data-opening');
      rest();
    };
    // by itself the last line fades before the page comes back, so the two are never on top of each
    // other; a visitor's own click gets the page at once
    if (event) back();
    else setTimeout(back, 300);
  };
  for (const name of ['pointerdown', 'keydown', 'wheel']) window.addEventListener(name, end, { capture: true, passive: true });

  app.setAttribute('data-opening', '');
  at(0.5, () => say(0));
  at(3.2, () => {
    say(-1);
    rig.look(wide, 2.4);              // the pull-back is the point: let it take its time
  });
  at(4.0, () => say(1));
  at(7.4, () => say(2));
  at(10.6, () => say(3));
  at(14.0, end);
  return { refresh };
}
