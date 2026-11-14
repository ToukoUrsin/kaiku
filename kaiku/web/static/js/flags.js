// What the address bar asks for. Everything is optional.
//
//   ?shot=lead|overview|top   camera to start on
//   ?env=K                    start close up on robot K
//   ?orbit=1                  turn slowly around the subject
//   ?clean=1                  for video: only the stage, the totals and a caption
//   ?sample=1                 fill the policy tab with sample numbers (marked as such)
//   ?tab=live|dataset|policy  panel tab to open
//   ?sheet=tall               phone: start with the panel pulled up
//   ?quality=high|low         pin the drawing quality instead of adapting to the frame rate
//   ?preview=NAME             show one state of the page without waiting for it (see sample.js)

const SHOTS = ['lead', 'overview', 'top'];
const TABS = ['live', 'dataset', 'policy'];

export function readFlags(search = location.search) {
  const params = new URLSearchParams(search);
  const on = (name) => params.has(name) && params.get(name) !== '0';
  const oneOf = (name, allowed, fallback) => (allowed.includes(params.get(name)) ? params.get(name) : fallback);
  const env = Number.parseInt(params.get('env') ?? '', 10);
  return {
    shot: oneOf('shot', SHOTS, 'lead'),
    env: Number.isInteger(env) && env >= 0 ? env : -1,
    orbit: on('orbit'),
    clean: on('clean'),
    sample: on('sample'),
    tab: oneOf('tab', TABS, 'live'),
    sheet: oneOf('sheet', ['tall'], ''),
    quality: oneOf('quality', ['high', 'low'], 'auto'),
    preview: params.get('preview') || '',
  };
}
