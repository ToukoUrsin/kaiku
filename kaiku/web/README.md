# Kaiku browser client

`server.py` owns the HTTP and WebSocket contract, `scene_export.py` what the page is told about
the model, `protocol.py` the binary frames. The page itself is plain ES modules in `static/`:
no build step, three.js vendored in `static/vendor`, fonts in `static/fonts`.

| File | What it does |
| --- | --- |
| `index.html` | All the markup. Scripts find elements by `data-ref`. |
| `css/tokens.css` | Colours, type scale, spacing. Green means a demonstration that succeeded and nothing else. |
| `css/app.css` | Layout and components, the phone sheet, `?clean=1`. |
| `js/main.js` | Boot (scene, meshes, stream), keys, and the frame loop. |
| `js/flags.js` | What the address bar can ask for. |
| `js/stream.js` | The WebSocket: frames, HUD, events, reconnecting. |
| `js/protocol.js` | Decoder for the quantised key and delta frames, and ACKs. |
| `js/scene.js` | `/api/scene` and the mesh packs, sorted into robot, parts and station furniture. |
| `js/fk.js` | Forward kinematics over the exported MuJoCo body tree. |
| `js/layout.js` | Where each robot stands: the operator's in front, the echoes in tiered ranks behind. |
| `js/stage.js` | Renderer, lights, floor, echo rings, quality steps. |
| `js/fleet.js` | Every robot, part and station as instanced meshes, with three levels of detail. |
| `js/markers.js` | The floor marker of each station: working, sorted, failed, the operator's. |
| `js/hands.js` | The operator's hands as a live skeleton over the robot they drive. |
| `js/camera.js` | Three shots, close-up of one robot, orbit, zoom, picking rays. |
| `js/hud.js` | Totals, panels, controls, banners, toasts: the HUD message written into the page. |
| `js/chart.js` | The policy chart. |
| `js/sample.js` | Sample policy numbers and state previews; used only when the address asks. |

Coordinates are MuJoCo's all the way to the screen: z up, the pelvis at the origin, +x forward,
+y left. Environment `k` is drawn at `layout.offsets[k]`; nothing is rotated in between.

## Address flags

`?shot=lead|overview|top`, `?env=K` (close up on robot K), `?orbit=1`, `?clean=1` (for video:
stage, totals and a caption), `?sample=1` (sample policy numbers, marked on screen),
`?tab=live|dataset|policy`, `?sheet=tall` (phone), `?quality=high|low`.

`?preview=NAME` shows one state without waiting for it: `loading`, `error`, `connecting`,
`reconnecting`, `paused`, `nogloves`, `gloves`, `autopilot`, `first` (nothing collected yet),
`training`, `partial` (a HUD with almost nothing in it), `toast`.

Keys: `1` `2` `3` shots, `Esc` back from a robot, arrows to the next robot, `Space` pause, `N`
next scene. In the console, `kaiku` holds the stage, fleet, stream and the rest.

## Test bed

`python -m kaiku.web.demo --n 64 --port 8650` serves the page with a synthetic stream, without the
runner. `--model twin` uses the challenge twin when it is importable.
