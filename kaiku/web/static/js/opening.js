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
 */
export function playOpening({ app, line, rig, wide, words, rest }) {
  const timers = [];
  let over = false;
  const at = (seconds, run) => timers.push(setTimeout(run, seconds * 1000));
  const say = (text) => {
    line.removeAttribute('data-show');
    if (!text) return;
    timers.push(setTimeout(() => {
      line.textContent = text;
      line.setAttribute('data-show', '');
    }, 280));
  };
  const end = () => {
    if (over) return;
    over = true;
    timers.forEach(clearTimeout);
    for (const name of ['pointerdown', 'keydown', 'wheel']) window.removeEventListener(name, end, true);
    line.removeAttribute('data-show');
    app.removeAttribute('data-opening');
    rest();
  };
  for (const name of ['pointerdown', 'keydown', 'wheel']) window.addEventListener(name, end, { capture: true, passive: true });

  app.setAttribute('data-opening', '');
  at(0.5, () => say(words()[0]));
  at(3.2, () => {
    say('');
    rig.look(wide, 2.4);              // the pull-back is the point: let it take its time
  });
  at(4.0, () => say(words()[1]));
  at(7.4, () => say(words()[2]));
  at(10.6, () => say(words()[3]));
  at(14.0, end);
}
