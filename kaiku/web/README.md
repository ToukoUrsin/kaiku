# Kaiku browser viewer

`server.py` owns the HTTP/WebSocket contract and scene export. The static client is split into:

- `static/js/protocol.js` — quantised key/delta frame decoder and ACKs.
- `static/js/fk.js` — forward kinematics over the exported MuJoCo body tree.
- `static/js/app.js` — three.js scene, instanced robot fleet, camera interaction and HUD panels.
- `static/css/app.css` and `static/index.html` — the responsive shell.

Run the self-contained test bed with `python -m kaiku.web.demo --n 64 --port 8650`, then open the printed URL. `--model twin` uses the challenge twin when it is importable.
