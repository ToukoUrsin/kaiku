# One Smartglove sitting

1. Bring one or two Rokoko Smartgloves II, optionally a Coil Pro, the receiver, and their charging cables.
2. Plug in the receiver and optional Coil Pro, put on the glove or gloves you brought, turn them on, and wait 20 seconds.
3. From `~/kaiku`, run `python -m kaiku.app.gloves`.
4. Wait while the terminal checklist stays on screen; it starts the page as soon as every line is ready.
5. On the page, hold both hands flat and still over the Coil Pro until calibration completes.
6. Sort the scenes the page gives you until it says **30 of 30 scenes**; each takes about 20–40 seconds, so the sitting is under 25 minutes.
7. Press Ctrl-C when the page says you can stop; the recordings are kept and the folder gets a `DONE` file.
8. On the box, run `python -m kaiku.app.sitting ~/kaiku-sessions/YYYYMMDD-HHMM`; it validates, makes the replay bundle, runs the study and films off-box, and writes `SITTING.md`.

| Terminal or page message | What to do |
|---|---|
| SDK is not installed | Install Rokoko Device SDK on this Mac, then run the one command again. |
| Not signed in | Open Rokoko and sign in on this Mac. |
| SDK is not activated | Activate the SDK on this Mac. |
| Licence expired or no seat | Renew the licence or free a seat. |
| Driver is not streaming | Leave the Rokoko driver running; if the port is open with no frames for 3 seconds, start streaming. |
| One glove | This is a valid setup: its fingers and pinch drive the hand; without a Coil Pro, the laptop camera supplies position. |
| Both gloves missing | Wake at least one glove and check its receiver and cable. |
| No Coil Pro | This is allowed; the laptop camera supplies hand position. |
| Rate too low | Move closer to the receiver and remove nearby radio interference. |
| Port 8651 is already taken | Kaiku chooses the next free port and prints its new address. |
| Scene dropped: glove stream stopped | Keep the glove awake; that scene was discarded and will not train the policy. |
| 30 of 30 scenes | Stop with Ctrl-C. |

The tested paths are Rokoko's driver with emulated Smartgloves II and Kaiku's own moving emulator. Real
Smartgloves II and Coil Pro hardware have not yet been tested on this box.
